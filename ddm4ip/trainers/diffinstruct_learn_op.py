import os
import warnings

from omegaconf import DictConfig
import torch

from ddm4ip.data.base import Batch, Datasplit, init_dataset
from ddm4ip.losses import DiffInstructOnY
from ddm4ip.networks import init_kernel_net
from ddm4ip.networks.blur_nets import DegradationNetwork
from ddm4ip.networks.unets import init_unet_with_defaults
from ddm4ip.trainers.base import BaseTrainer, init_trts_dataloaders
from ddm4ip.trainers.plots import plot_explicit_kernels, plot_imggen, plot_implicit_kernels, plot_spacevarying_kernels
from ddm4ip.utils import distributed
from ddm4ip.utils.easy_dict import print_module_summary


class DiffinstructOpTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.plot_batch = None
        self.corrupt_img_size: tuple[int, int, int] | None = None

    def init_datasets(self, cfg: DictConfig):
        if self.is_training:
            return {
                Datasplit.TRAIN: init_dataset(cfg, split=Datasplit.TRAIN, is_paired=False), # type: ignore
                Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True) # type: ignore
            }
        return {
            Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True) # type: ignore
        }

    def init_dataloaders(self, cfg, dsets):
        # No infinite data-loader during evaluation
        return init_trts_dataloaders(
            cfg, dsets, start_idx=self.start_global_step, is_infinite=self.is_training
        )

    def _init_pretrained_unet(self, config, img_size, cond_ch, label_dim):
        """A frozen teacher loaded from a stage-1 checkpoint (its EMA weights), and
        an auxiliary network with the same architecture initialized from it."""
        if "path" not in config:
            raise ValueError(
                "The pretrained flow configuration must contain a "
                "'path' key pointing to the pretrained model checkpoint."
            )
        if not os.path.isfile(prt_path := config["path"]):
            raise FileNotFoundError(
                f"Pretrained flow model path '{prt_path}' is not a file."
            )
        ckpt = torch.load(prt_path, map_location="cpu", weights_only=False)
        state = ckpt["ema"]["emas"][-1]
        print(f"Loading pretrained flow model. EMA state with std: {ckpt['ema']['stds'][-1]}")

        unet = init_unet_with_defaults(
            cfg=config, img_size=img_size, cond_ch=cond_ch, label_dim=label_dim,
        )
        unet.load_state_dict(state)
        aux_unet = init_unet_with_defaults(
            cfg=config, img_size=img_size, cond_ch=cond_ch, label_dim=label_dim,
        )
        aux_unet.load_state_dict(state)
        unet.requires_grad_(False)
        return unet, aux_unet

    def init_models(self, cfg, dsets, device):
        dataset = self.get_any_dset(dsets)
        assert dataset is not None

        kernel_nn = init_kernel_net(cfg["models"]["kernel"]).to(device)
        kernel_nn_fake_inputs = {
            "clean_img": torch.zeros([self.batch_size, *dataset.clean_img_size], device=device),
            "sigma": torch.zeros([self.batch_size], device=device),
            "conditioning": torch.zeros([self.batch_size, dataset.clean_conditioning_channels, *dataset.clean_img_size[1:]], device=device)
        }
        kernel_nn_fake_output: torch.Tensor = kernel_nn(**kernel_nn_fake_inputs)
        self.corrupt_img_size = (kernel_nn_fake_output.shape[1], kernel_nn_fake_output.shape[2], kernel_nn_fake_output.shape[3])

        # Teacher and auxiliary network on the degraded images.
        flow_input_size = self.corrupt_img_size
        cond_ch = dataset.corrupt_conditioning_channels
        prtr_flow_nn, aux_flow_nn = self._init_pretrained_unet(
            cfg["models"]["pretrained_flow"],
            img_size=flow_input_size,
            cond_ch=cond_ch,
            label_dim=dataset.label_dim,
        )
        prtr_flow_nn.to(device)
        aux_flow_nn.to(device)

        if distributed.get_rank() == 0:
            print("-----------------------AUXILIARY FLOW NN------------------------")
            print_module_summary(aux_flow_nn, {
                "x": torch.zeros([self.batch_size, *flow_input_size], device=device),
                "sigma": torch.ones([self.batch_size], device=device),
                "conditioning": torch.zeros([self.batch_size, cond_ch, *flow_input_size[1:]], device=device)
            }, max_nesting=2)
            print("-------------------------KERNEL NN-------------------------")
            print_module_summary(kernel_nn, kernel_nn_fake_inputs, max_nesting=2)

        return {
            "kernel_nn": kernel_nn,
            "prtr_flow_nn": prtr_flow_nn,
            "aux_flow_nn": aux_flow_nn,
        }

    def init_loss(self, cfg, models):
        match cfg["loss"].get("name"):
            case None:
                raise ValueError("Could not instantiate un-named loss.")
            case "diy":
                return DiffInstructOnY(cfg, **models)  # pyright: ignore[reportArgumentType]
            case unsupported:
                raise ValueError(f"Loss name {unsupported} invalid. Expected 'diy'.")

    def validate_batch(self, val_batch):
        return self.loss_optim.val_loss(self, val_batch)

    def get_any_dset(self, dsets):
        return next(d for d in dsets.values() if d is not None)

    def make_plots(self, cfg, dsets, dloaders, models, tb_writer, global_step):
        if self.plot_batch is None:
            # This ensures we always use the same batch for plotting.
            if (ds := dsets.get(Datasplit.TEST)) is None:
                warnings.warn("Using training batch for plotting.")
                ds = dsets[Datasplit.TRAIN]
                assert ds is not None
            # Don't use the dataloader to get around data-sampling artefacts.
            # messing around with indexes to get decent images for plotting
            self.plot_batch = Batch.collate_fn([ds[i] for i in range(min(self.batch_size, len(ds)))])

        kernel_nn = models["kernel_nn"]
        assert isinstance(kernel_nn, DegradationNetwork)
        dset = self.get_any_dset(dsets)
        assert self.corrupt_img_size is not None

        p1 = None
        if "aux_flow_nn" in models:
            p1 = plot_imggen(cfg, models["aux_flow_nn"], self.plot_batch, self.corrupt_img_size, dset, self.device)
        if p1 is not None and distributed.get_rank() == 0:
            self.log_image(p1, 'diffusion/generated', tb_writer)

        p2, p2_ker = plot_spacevarying_kernels(
            dset.corruption, dset.corrupt_img_size[-1], kernel_nn, self.batch_size, self.device
        )
        if p2 is None:
            p2, p2_ker = plot_explicit_kernels(dset.corruption, kernel_nn, self.plot_batch, self.batch_size, self.device)
        if p2 is not None and distributed.get_rank() == 0:
            self.log_image(p2, 'eval/explicit-kernels', tb_writer)
        if p2_ker is not None and distributed.get_rank() == 0:
            tb_writer.add_tensor('eval/kernel', p2_ker, self.global_step)
        # With centering, also log the kernel before the sub-pixel recentering.
        if getattr(kernel_nn.physics_net, "auto_center", False):
            p2raw, _ = plot_explicit_kernels(
                dset.corruption, kernel_nn, self.plot_batch,
                self.batch_size, self.device, center=False,
            )
            if p2raw is not None and distributed.get_rank() == 0:
                self.log_image(p2raw, 'eval/explicit-kernels-raw', tb_writer)
        p3 = plot_implicit_kernels(self.plot_batch, kernel_nn, loss=self.loss_optim)
        if p3 is not None and distributed.get_rank() == 0:
            self.log_image(p3, 'eval/implicit-kernels', tb_writer)
