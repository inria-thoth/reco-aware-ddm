import math
import os
import warnings

from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
from torchvision.utils import make_grid
import deepinv as dinv

from ddm4ip.data.base import Batch, Datasplit, init_dataset
from ddm4ip.data.data2 import KnownCorruptionDataset
from ddm4ip.losses.deepinv_loss import DeepInvLoss
from ddm4ip.networks import init_kernel_net
from ddm4ip.networks.blur_nets import DegradationNetwork
from ddm4ip.trainers.base import BaseTrainer, init_trts_dataloaders
from ddm4ip.trainers.plots import plot_spacevarying_kernels
from ddm4ip.utils import distributed
from ddm4ip.utils.torch_utils import write_img_pt


from ddm4ip.pnp.dpir import DPIRSolver


def _ema_kernel_state(pretr_data):
    """State dict of the operator EMA stored in a stage-2 checkpoint, or None."""
    loss_state = pretr_data.get("loss_optim") or {}
    keb = loss_state.get("kernel_ema_beta")
    if keb is not None and isinstance(keb, dict) and keb.get("ema") is not None:
        return keb["ema"]
    return None


class DeepinvDenoiserTrainer(BaseTrainer):
    """
    Stage 3: restores a paired test set with DPIR, using either the operator
    learned at stage 2 (models.kernel.path), a kernel read from a .mat file
    (models.kernel.fixed_kernel_path, super-resolution), or the ground-truth PSF
    grid (models.kernel.initialization=from_psf, space-varying oracle).
    """
    def __init__(self):
        super().__init__()
        self.plot_dloader = None
        self.physics = None

    def init_datasets(self, cfg: DictConfig):
        return {
            # This dataset should be paired!
            Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True) # pyright: ignore[reportArgumentType]
        }

    def init_dataloaders(self, cfg, dsets):
        # No infinite data-loader because we want to loop through the test
        # dataset once only.
        return init_trts_dataloaders(
            cfg, dsets, start_idx=self.start_global_step, is_infinite=False
        )

    def init_dpir(self, prior_model, sigma, invp_cfg=None):
        invp_cfg = invp_cfg or {}
        self.restorer = DPIRSolver(
            noise_level=sigma,
            grad_iters=0,
            prior_model=prior_model,
            max_iter=invp_cfg.get("max_iter", 8),
            sat_mask=invp_cfg.get("sat_mask", False),
            sat_hi=invp_cfg.get("sat_hi", 0.99),
            sat_weight=invp_cfg.get("sat_weight", 0.05),
            sat_target=invp_cfg.get("sat_target", 1.0),
        )
        return self.restorer

    def init_deep_prior(self, network: str, device, large_model: bool = False):
        if network.lower() == "drunet":
            model = dinv.models.DRUNet(pretrained="download").to(device=device)
        elif network.lower() == "diffunet":
            model = dinv.models.DiffUNet(large_model=large_model, use_fp16=False, pretrained="download").to(device=device)
        else:
            raise ValueError(f"Network '{network}' is not a valid deep prior.")
        distributed.print0(f"Initialized {network} prior")
        return model

    def init_models(self, cfg, dsets, device):
        models = {}
        test_dset = dsets[Datasplit.TEST]
        assert test_dset is not None
        fixed_kernel_path = cfg.models.kernel.get("fixed_kernel_path")
        if fixed_kernel_path:
            # Super-resolution with a supplied kernel (scipy .mat, key "Kernel"),
            # applied through a size-reducing operator so that DPIR solves the
            # LR->HR problem directly.
            import scipy.io
            k = torch.as_tensor(scipy.io.loadmat(fixed_kernel_path)["Kernel"]).float()
            while k.ndim < 4:           # -> (1, C, kh, kw)
                k = k.unsqueeze(0)
            k = k.clamp_min(0)
            k = k / k.flatten(-2).sum(-1, keepdim=True).clamp_min(1e-8).unsqueeze(-1)
            pad = cfg.models.kernel.get("padding", "circular")
            sr_factor = int(cfg.models.kernel.get("sr_factor", 1) or 1)
            if sr_factor <= 1:
                raise ValueError("models.kernel.fixed_kernel_path requires models.kernel.sr_factor > 1")
            if pad == "zeros_normalized":
                # Normalized convolution (see NormalizedDownsampling); deepinv's
                # prox_l2 falls back to conjugate gradient for this operator.
                from ddm4ip.networks.blur_nets import NormalizedDownsampling
                physics = NormalizedDownsampling(
                    img_size=None, filter=k.to(device),
                    factor=sr_factor, device=device)
            else:
                from ddm4ip.degradations.blur import Downsampling
                physics = Downsampling(img_size=None, filter=k.to(device),
                                       factor=sr_factor, padding=pad, device=device)
            print(f"Loaded fixed kernel {tuple(k.shape)} as Downsampling "
                  f"(factor={sr_factor}, padding={pad}) from '{fixed_kernel_path}'.")
            models["kernel_nn"] = physics
        elif (pert_path := cfg.models.kernel.get("path")) is not None:
            # The stage-2 operator; with use_ema, its EMA copy. Centering is
            # applied as during training through the auto_center_* settings.
            use_ema = bool(cfg.models.kernel.get("use_ema", False))
            if not pert_path.endswith(".pt"):
                raise ValueError(
                    "models.kernel.path must be a stage-2 training-state-*.pt "
                    f"checkpoint, got '{pert_path}'")
            kernel_nn = init_kernel_net(cfg["models"]["kernel"])
            pretr_data = torch.load(pert_path, map_location="cpu", weights_only=False)
            state = pretr_data["kernel_nn"]["state_dict"]
            if use_ema:
                ema_state = _ema_kernel_state(pretr_data)
                if ema_state is not None:
                    state = ema_state
                    print(f"Loading EMA kernel from '{pert_path}'.")
                else:
                    print(f"WARNING: models.kernel.use_ema=true but no EMA "
                          f"kernel in '{pert_path}'; using the raw kernel.")
            kernel_nn.load_state_dict(state)
            models["kernel_nn"] = kernel_nn.to(device)
        elif str(cfg.models.kernel.get("initialization", "")) == "from_psf":
            # Oracle: a frozen kernel net whose grid is the ground-truth PSF grid,
            # so that DPIR restores each patch with the true kernel at its centre.
            kernel_nn = init_kernel_net(cfg["models"]["kernel"])
            print("Oracle: kernel net built from the ground-truth PSF grid.")
            models["kernel_nn"] = kernel_nn.to(device)
        else:
            raise ValueError("Stage 3 needs models.kernel.path, models.kernel.fixed_kernel_path "
                             "or models.kernel.initialization=from_psf.")

        noise_params = self.get_noise_params(test_dset, models["kernel_nn"])
        invp_cfg = cfg["models"]["deepinv_solver"]
        if "sigma" not in noise_params:
            # Noise-free measurements (super-resolution): DPIR regularization level.
            noise_params["sigma"] = 0.01
            print("No noise sigma; using DPIR sigma=0.01.")
        solver_method = invp_cfg["method"]
        if solver_method.lower() != "dpir":
            raise ValueError(f"deepinv_solver method '{solver_method}' is invalid; only 'dpir' is supported.")
        prior = self.init_deep_prior(invp_cfg["prior"], device, large_model=invp_cfg.get("large_model", False))
        models["deepinv_model"] = self.init_dpir(prior, invp_cfg=invp_cfg, **noise_params)

        return models

    def get_noise_params(self, test_dset, kernel_model) -> dict[str, float]:
        fixed_sigma = None
        if not isinstance(kernel_model, dinv.physics.Physics):
            assert isinstance(kernel_model, DegradationNetwork), f"Expected kernel model of type DegradationNetwork, found {type(kernel_model)}"
            fixed_sigma = getattr(kernel_model.noise_net, "fixed_sigma", None)
        if fixed_sigma is not None:
            # The forward noise level the operator was trained with.
            noise_params = {"sigma": fixed_sigma}
            print(f"Using the run's fixed forward sigma ({float(fixed_sigma):.5g}).")
        else:
            # Otherwise the noise level is assumed known and read from the dataset.
            assert isinstance(test_dset, KnownCorruptionDataset)
            noise_params = test_dset.noise_params
        return {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in noise_params.items()
        }

    def init_loss(self, cfg, models) -> DeepInvLoss:
        return DeepInvLoss(
            cfg,
            deepinv_model=models["deepinv_model"],
            kernel_nn=models["kernel_nn"]
        )

    def validate_batch(self, val_batch: Batch):
        metrics, pred_img, filters = self.loss_optim.val_loss_with_output(self, val_batch)
        if self.save_eval_to_file:
            assert self.eval_dir is not None and val_batch.corrupt is not None and val_batch.clean is not None
            if self.save_pred_only:
                img_batch = pred_img.cpu()
            elif val_batch.corrupt.shape == val_batch.clean.shape:
                img_batch = torch.cat([val_batch.corrupt.cpu(), pred_img.cpu(), val_batch.clean.cpu()], dim=-1)
            else:
                img_batch = torch.cat([pred_img.cpu(), val_batch.clean.cpu()], dim=-1)
            if filters is not None:
                # Filters are just saved as pt, since it's too hard to convert them into good
                # looking images for all possible situations.
                torch.save(filters, os.path.join(self.eval_dir, f"kernel_{self.global_step:05d}.pt"))
            assert img_batch.dim() == 4
            for i in range(img_batch.shape[0]):
                write_img_pt(img_batch[i], os.path.join(self.eval_dir, f"img_{self.global_step + i:05d}.png"))
            if self.save_corrupt:
                # The degraded inputs, index-aligned with the img_*.png restorations.
                corrupt_batch = val_batch.corrupt.cpu()
                for i in range(corrupt_batch.shape[0]):
                    write_img_pt(corrupt_batch[i], os.path.join(self.eval_dir, f"orig_{self.global_step + i:05d}.png"))
        return metrics

    def make_plots(self, cfg, dsets, dloaders, models, tb_writer, global_step):
        if self.plot_dloader is None:
            if (dl := dloaders.get(Datasplit.TEST)) is None:
                warnings.warn("Using training batch for plotting.")
                dl = dloaders[Datasplit.TRAIN]
                assert dl is not None
            self.plot_dloader = iter(dl)

        try:
            plot_batch = next(self.plot_dloader)
        except StopIteration:
            return

        models["deepinv_model"].eval()
        testset = dsets[Datasplit.TEST]
        assert testset is not None

        p1 = self.plot_generated(plot_batch, models["deepinv_model"])
        if distributed.get_rank() == 0 and p1 is not None:
            self.log_image(p1, 'denoiser/generated', tb_writer)

        p2 = None
        try:
            p2, p2_ker = plot_spacevarying_kernels(
                testset.corruption,
                self.loss_optim.patch_size or testset.corrupt_img_size[-1],
                self.models["kernel_nn"],
                self.batch_size,
                self.device
            )
        except TypeError:
            pass
        if distributed.get_rank() == 0 and p2 is not None:
            self.log_image(p2, 'denoiser/kernels', tb_writer)

    @torch.no_grad()
    def plot_generated(self, batch: Batch, deepinv_model, num_imgs: int = 16):
        dev_batch = batch[:num_imgs].to(self.device)
        if dev_batch.clean is None:
            return None
        assert dev_batch.corrupt is not None
        target = dev_batch.clean
        source = dev_batch.corrupt

        preds, _ = self.loss_optim.run_model(dev_batch)

        batch_size = target.shape[0]
        n_img_row = int(math.sqrt(batch_size))

        target = make_grid(
            target.clamp(0, 1),
            nrow=n_img_row, padding=0,
        ).permute(1, 2, 0).numpy(force=True)
        source = make_grid(
            source.clamp(0, 1),
            nrow=n_img_row, padding=0,
        ).permute(1, 2, 0).numpy(force=True)
        preds = make_grid(
            preds.clamp_(0, 1),
            nrow=n_img_row, padding=0,
        ).permute(1, 2, 0).numpy(force=True)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))

        ax[0, 0].imshow(source, aspect='equal', interpolation='none')
        ax[0, 0].set_title(r"noisy")
        ax[0, 0].set_axis_off()

        ax[0, 1].imshow(target, aspect='equal', interpolation='none')
        ax[0, 1].set_title(r"clean")
        ax[0, 1].set_axis_off()

        ax[1, 0].imshow(preds, aspect='equal', interpolation='none')
        ax[1, 0].set_title(r"predicted")
        ax[1, 0].set_axis_off()

        diff = (target - preds).mean(2)  # average over channels
        title = r"errors"
        if np.abs(diff).max() < 0.1:
            diff *= 10
            title = rf"{title} (x10)"
        ax[1, 1].imshow(diff, aspect='equal', interpolation='none', cmap='RdBu', vmin=-1.0, vmax=1.0)
        ax[1, 1].set_title(title)
        ax[1, 1].set_axis_off()

        fig.tight_layout()
        return fig
