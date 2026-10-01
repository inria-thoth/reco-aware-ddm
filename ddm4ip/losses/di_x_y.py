from typing import Any, Mapping
import warnings

import torch
import torch.nn as nn

from ddm4ip.data.base import Batch, Datasplit
from ddm4ip.losses.base import AbstractLoss, OptState, OptimizationState
from ddm4ip.losses.flow_matching import FlowMatchingLoss
from ddm4ip.losses.utils import create_optimizer, make_ddp
from ddm4ip.networks.blur_nets import DegradationNetwork
from ddm4ip.trainers.base import BaseTrainer
from ddm4ip.pnp import dpir
from ddm4ip.utils.phema import FixedBetaEMA
from ddm4ip.utils.torch_utils import (
    center,
    equate_kernel_shapes,
    shift_invariant_kernel_mae,
)


class DiffInstructOnXandY(AbstractLoss):
    """Measurement term F (distribution matching on y) plus reconstruction term G
    (distribution matching on DPIR restorations), combined with gradient-norm
    balancing and a linear warm-up of lambda."""

    GRAD_BALANCE_EPS = 1e-12

    def __init__(
        self,
        config: Mapping[str, Any],
        aux_flow_nn_x: nn.Module,
        prtr_flow_nn_x: nn.Module,
        aux_flow_nn_y: nn.Module,
        prtr_flow_nn_y: nn.Module,
        kernel_nn: DegradationNetwork,
    ):
        loss_cfg = config['loss']
        rest_cfg = loss_cfg['restorer']
        super().__init__(has_val_loss=True, n_accum_steps=loss_cfg['n_accum_steps'])

        self.opt_state = OptimizationState(0, OptState.INNER)
        self.kernel_opt, self.lr_scheduler = create_optimizer(
            config['optim']['kernel'], kernel_nn.parameters()
        )
        gs_enabled = getattr(kernel_nn, "use_fp16", False)
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=gs_enabled)

        # torch.compile the four flow networks in place (identity is preserved for
        # FlowMatchingLoss / DDP). The recompile limit is raised above the number
        # of distinct UNet block shapes so that no block falls back to eager mode.
        if bool(loss_cfg.get('compile', False)):
            import torch._dynamo as _dynamo
            for _attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(_dynamo.config, _attr):
                    setattr(_dynamo.config, _attr, 64)
            aux_flow_nn_x.compile(dynamic=True)
            prtr_flow_nn_x.compile(dynamic=True)
            aux_flow_nn_y.compile(dynamic=True)
            prtr_flow_nn_y.compile(dynamic=True)

        aux_cfg_x = dict(loss=loss_cfg['aux_x'], optim=config['optim']['aux_x'])
        aux_cfg_x['n_accum_steps'] = self.n_accum_steps
        aux_cfg_y = dict(loss=loss_cfg['aux_y'], optim=config['optim']['aux_y'])
        aux_cfg_y['n_accum_steps'] = self.n_accum_steps

        self.aux_flow_loss_x = FlowMatchingLoss(
            config=aux_cfg_x, flow_nn=aux_flow_nn_x, clean_images=True
        )
        self.aux_flow_nn_x = aux_flow_nn_x
        self.prtr_flow_nn_x = prtr_flow_nn_x.requires_grad_(False)

        self.aux_flow_loss_y = FlowMatchingLoss(
            config=aux_cfg_y, flow_nn=aux_flow_nn_y, clean_images=True
        )
        self.aux_flow_nn_y = aux_flow_nn_y
        self.prtr_flow_nn_y = prtr_flow_nn_y.requires_grad_(False)

        self.kernel_nn = kernel_nn
        self.kernel_nn_ddp = make_ddp(kernel_nn, broadcast_buffers=False)
        self._kernel_params = [p for p in kernel_nn.parameters() if p.requires_grad]

        # Fixed-decay EMA of the operator; validation and evaluation use its copy.
        self.kernel_ema_beta = None
        _beta = loss_cfg.get('ema_kernel_beta', None)
        if _beta is not None:
            self.kernel_ema_beta = FixedBetaEMA(kernel_nn, beta=float(_beta))

        # Differentiable DPIR restorer (DRUNet prior); only the last
        # `grad_iters` iterations are differentiated.
        if rest_cfg.get('type', 'dpir') != 'dpir':
            raise ValueError(f"Restorer type {rest_cfg['type']} is invalid; only 'dpir' is supported.")
        self.restorer = dpir.DPIRSolver(
            prior_model=None,
            noise_level=rest_cfg['sigma'],
            grad_iters=rest_cfg['grad_iters'],
            max_iter=rest_cfg.get('max_iter', 8),
            compile=rest_cfg.get('compile', False),
        )
        self.restorer.to(device=torch.cuda.current_device()).requires_grad_(False)

        # lambda(k) = min(1, k / K_warm) * lambda
        self.lambda_dix = float(loss_cfg.get('lambda_dix', 1.0))
        warmup = loss_cfg.get('lambda_dix_warmup_steps', None)
        self.lambda_dix_warmup_steps = int(warmup) if warmup is not None else None

    def _current_lambda_dix(self, global_step: int) -> float:
        if self.lambda_dix_warmup_steps is None or self.lambda_dix_warmup_steps <= 0:
            return self.lambda_dix
        frac = min(1.0, max(0.0, float(global_step) / self.lambda_dix_warmup_steps))
        return frac * self.lambda_dix

    @staticmethod
    def _grads_l2(grads) -> float:
        """L2 norm over a list of (possibly None) gradient tensors."""
        sq = 0.0
        for g in grads:
            if g is not None:
                sq = sq + g.detach().float().pow(2).sum()
        if isinstance(sq, float):
            return 0.0
        return float(sq.sqrt().item())

    def get_images(self, kernel_nn: torch.nn.Module, start_img: torch.Tensor, return_physics=False, **kwargs):
        _, physics = kernel_nn(
            start_img,
            return_physics=True,
            **kwargs
        )
        out = self.restorer(start_img, physics)
        if return_physics:
            return out, physics
        return out

    def _dix_surrogate(self, x_hat: torch.Tensor, conditioning) -> torch.Tensor:
        """Distribution-matching surrogate on restored images. The velocity
        difference is computed without gradient, so the operator gradient flows
        only through x_hat (hence through the restorer)."""
        diff = self.aux_flow_loss_x.diffusion
        x1 = center(x_hat)
        x0 = diff.sample_noisy(x1)
        t = diff.sample_time(x0.shape[0], x0.device)
        xt = diff.make_xt(x_clean=x1, x_noise=x0, t=t).to("cuda")

        self.prtr_flow_nn_x.train()
        self.aux_flow_nn_x.train()
        with torch.no_grad():
            # Same dropout masks for the teacher and the auxiliary field.
            cuda_rng_state = torch.cuda.get_rng_state()
            prtr_velocity = self.prtr_flow_nn_x(xt, t, conditioning=conditioning)
            torch.cuda.set_rng_state(cuda_rng_state)
            aux_velocity = self.aux_flow_nn_x(xt, t, conditioning=conditioning)
            flow_diff = aux_velocity - prtr_velocity
        self.prtr_flow_nn_x.eval()

        return (flow_diff * x1).mean()

    def loss_di_x(self, noisy_img: torch.Tensor, conditioning, **kernel_kwargs):
        x_hat_real, _ = self.get_images(
            self.kernel_nn_ddp,
            noisy_img,
            return_physics=True,
            conditioning=conditioning,
            **kernel_kwargs
        )
        loss_di = self._dix_surrogate(x_hat_real, conditioning)
        loss_info = {"loss/di_x": loss_di.detach()}
        # The restoration is returned so that the outer step can reuse it
        # (detached) to train the auxiliary field on x.
        return loss_di, loss_info, x_hat_real

    def loss_di_y(self, clean_img: torch.Tensor, conditioning, **kernel_kwargs):
        diff = self.aux_flow_loss_y.diffusion
        y1 = self.kernel_nn_ddp(
            clean_img,
            return_physics=False,
            conditioning=conditioning,
            **kernel_kwargs
        )
        y1 = center(y1)
        y0 = diff.sample_noisy(y1)
        t = diff.sample_time(y0.shape[0], y0.device)
        yt = diff.make_xt(x_clean=y1, x_noise=y0, t=t)

        with torch.no_grad():
            cuda_rng_state = torch.cuda.get_rng_state()
            prtr_velocity = self.prtr_flow_nn_y(yt, t, conditioning=conditioning)
            torch.cuda.set_rng_state(cuda_rng_state)
            aux_velocity = self.aux_flow_nn_y(yt, t, conditioning=conditioning)
            flow_diff = aux_velocity - prtr_velocity
            # Label-free monitor: fm_diff = fm_prtr - fm_aux >= 0, -> 0 as the
            # simulated measurements match the real ones.
            fm_target = y1 - y0
            fm_prtr = ((prtr_velocity - fm_target) ** 2).mean()
            fm_aux = ((aux_velocity - fm_target) ** 2).mean()
            fm_diff = fm_prtr - fm_aux

        loss_di = (flow_diff * y1).mean()
        loss_info = {
            "loss/di_y": loss_di.detach(),
            "loss/fm_prtr": fm_prtr,
            "loss/fm_aux": fm_aux,
            "loss/fm_diff": fm_diff,
        }
        return loss_di, loss_info

    def __call__(self, trainer: BaseTrainer, batch: Batch):
        batch_dev = batch.cuda()
        assert batch_dev.corrupt is not None and batch_dev.clean is not None
        self.kernel_nn_ddp.train()

        if self.opt_state.state == OptState.OUTER:  # operator update
            with self.optimizer_step(self.kernel_opt, self.lr_scheduler, self.kernel_nn_ddp, grad_scaler=self.grad_scaler):
                cur_lambda_dix = self._current_lambda_dix(trainer.global_step)
                # While lambda(k) = 0 the reconstruction branch is not evaluated.
                x_hat_real = None
                if cur_lambda_dix > 0:
                    loss_di_x, loss_info, x_hat_real = self.loss_di_x(
                        batch_dev.corrupt,
                        batch_dev.corrupt_conditioning,
                        **batch_dev.noise_params,
                        **batch_dev.meta
                    )
                else:
                    loss_di_x = torch.tensor(0.0, device=batch_dev.corrupt.device)
                    loss_info = {"loss/di_x": loss_di_x.detach()}
                loss_di_y, loss_info_y = self.loss_di_y(
                    batch_dev.clean,
                    batch_dev.clean_conditioning,
                    **batch_dev.noise_params,
                    **batch_dev.meta
                )
                loss_info.update(loss_info_y)

                # Gradient-norm balancing:
                #   g = g_F + lambda(k) * min(1, ||g_F|| / (||g_G|| + eps)) * g_G
                inv_accum = 1.0 / self.n_accum_steps
                diy_grads = torch.autograd.grad(
                    loss_di_y, self._kernel_params, allow_unused=True,
                )
                g_diy = self._grads_l2(diy_grads)
                if cur_lambda_dix > 0:
                    dix_grads = torch.autograd.grad(
                        loss_di_x, self._kernel_params, allow_unused=True,
                    )
                    g_dix = self._grads_l2(dix_grads)
                    scale = cur_lambda_dix * min(
                        1.0, g_diy / (g_dix + self.GRAD_BALANCE_EPS)
                    )
                else:
                    dix_grads = None
                    g_dix = 0.0
                    scale = 0.0
                for i, p in enumerate(self._kernel_params):
                    contrib = diy_grads[i]
                    if dix_grads is not None and dix_grads[i] is not None:
                        term = scale * dix_grads[i]
                        contrib = term if contrib is None else contrib + term
                    if contrib is None:
                        continue
                    contrib = (contrib * inv_accum).detach()
                    p.grad = contrib if p.grad is None else p.grad + contrib
                loss_info["loss/grad_balance_scale"] = scale
                loss_info["loss/grad_norm_diy"] = g_diy
                loss_info["loss/grad_norm_dix"] = g_dix
                loss_info["loss/grad_norm_dix_eff"] = scale * g_dix
                loss_info["loss/grad_norm_ratio"] = (
                    scale * g_dix / (g_diy + 1e-12)
                )
                loss_info["loss/lambda_dix_eff"] = cur_lambda_dix

            loss_info["learning_rate/kernel_nn"] = self.kernel_opt.param_groups[0]['lr']
            # In-place integer recentering of the operator parameter (no-op
            # unless enabled), before the EMA update so the EMA tracks it.
            self.kernel_nn.recenter_kernel_inplace_(self.kernel_opt)
            if self.kernel_ema_beta is not None:
                self.kernel_ema_beta.update()
            # Once the reconstruction branch is active, the auxiliary field on x
            # is trained here on the (detached) restoration of the outer step,
            # so that DPIR runs once per cycle.
            if x_hat_real is not None:
                ker_batch = Batch(
                    clean=x_hat_real.detach(),
                    clean_conditioning=batch_dev.corrupt_conditioning,
                    corrupt=None, clean_label=None, corrupt_label=None,
                    corrupt_conditioning=None, noise_params={},
                )
                aux_loss_info_x = self.aux_flow_loss_x(trainer, ker_batch)
                loss_info.update({
                    k.replace("loss/", "loss/aux_x_"): v
                    for k, v in aux_loss_info_x.items()
                })
            if super().is_full_step_complete():
                self.opt_state.increment_state()
        else:  # auxiliary-field update
            loss_info = {}
            # During the warm-up, the auxiliary field on x is trained here so
            # that it is ready when the reconstruction branch starts.
            if not self._current_lambda_dix(trainer.global_step) > 0:
                with torch.no_grad():
                    self.kernel_nn.eval()
                    x1 = self.get_images(
                        self.kernel_nn,
                        batch_dev.corrupt,
                        conditioning=batch_dev.corrupt_conditioning,
                        **batch_dev.noise_params,
                        **batch_dev.meta
                    )
                ker_batch = Batch(
                    clean=x1,
                    clean_conditioning=batch_dev.corrupt_conditioning,
                    corrupt=None, clean_label=None, corrupt_label=None,
                    corrupt_conditioning=None, noise_params={},
                )
                aux_loss_info_x = self.aux_flow_loss_x(trainer, ker_batch)
                loss_info = {
                    k.replace("loss/", "loss/aux_x_"): v for k, v in aux_loss_info_x.items()
                }

            with torch.no_grad():
                self.kernel_nn.eval()
                y1 = self.kernel_nn(
                    batch_dev.clean,
                    return_physics=False,
                    conditioning=batch_dev.clean_conditioning,
                    **batch_dev.noise_params,
                    **batch_dev.meta
                )
            ker_batch = Batch(
                clean=y1,
                clean_conditioning=batch_dev.clean_conditioning,
                clean_label=None, corrupt=None, corrupt_label=None,
                corrupt_conditioning=None, noise_params={},
            )
            aux_loss_info_y = self.aux_flow_loss_y(trainer, ker_batch)
            aux_loss_info_y = {
                k.replace("loss/", "loss/aux_y_"): v for k, v in aux_loss_info_y.items()
            }

            loss_info.update(aux_loss_info_y)
            if self.aux_flow_loss_y.is_full_step_complete():
                self.opt_state.increment_state()

        return loss_info

    def is_full_step_complete(self) -> bool:
        # A full step is one inner and one outer update.
        return super().is_full_step_complete() and self.opt_state.state == OptState.INNER

    def state_dict(self):
        return {
            "aux_x": self.aux_flow_loss_x.state_dict(),
            "aux_y": self.aux_flow_loss_y.state_dict(),
            "kernel_opt": self.kernel_opt.state_dict(),
            "kernel_lr": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "gs": self.grad_scaler.state_dict(),
            "kernel_ema_beta": (
                self.kernel_ema_beta.state_dict()
                if self.kernel_ema_beta is not None else None
            ),
        }

    def load_state_dict(self, state_dict):
        self.aux_flow_loss_x.load_state_dict(state_dict["aux_x"])
        self.aux_flow_loss_y.load_state_dict(state_dict["aux_y"])
        self.kernel_opt.load_state_dict(state_dict["kernel_opt"])
        if self.lr_scheduler is not None:
            if "kernel_lr" not in state_dict:
                raise ValueError(
                    "Created loss with lr scheduler but state-dict has no lr scheduler state."
                )
            self.lr_scheduler.load_state_dict(state_dict["kernel_lr"])
        self.grad_scaler.load_state_dict(state_dict["gs"])
        if self.kernel_ema_beta is not None and state_dict.get("kernel_ema_beta") is not None:
            self.kernel_ema_beta.load_state_dict(state_dict["kernel_ema_beta"])

    @torch.no_grad()
    def val_loss(self, trainer: BaseTrainer, batch: Batch) -> Mapping[str, float | torch.Tensor]:
        vl = {}
        # Here we assume batch contains paired data
        dev_batch = batch.cuda()
        assert dev_batch.corrupt is not None
        kernel_net_for_val = (
            self.kernel_ema_beta.ema
            if self.kernel_ema_beta is not None
            else self.kernel_nn
        )
        pred_clean, physics = self.get_images(
            kernel_net_for_val,
            dev_batch.corrupt,
            return_physics=True,
            conditioning=dev_batch.corrupt_conditioning,
            **dev_batch.noise_params,
            **dev_batch.meta
        )
        if dev_batch.clean is not None and dev_batch.clean.shape == pred_clean.shape:
            vl["val_loss/x-mae"] = torch.mean(torch.abs(dev_batch.clean - pred_clean))
        else:
            warnings.warn("Cannot compute validation loss on x.")

        try:
            if hasattr(physics, "filter") and physics.filter is not None:
                if dev_batch.kernel is not None:
                    true_kernel = dev_batch.kernel
                elif Datasplit.TRAIN in trainer.dsets and hasattr(trainer.dsets[Datasplit.TRAIN].corruption, 'get_kernel'): # pyright: ignore[reportOptionalMemberAccess]
                    true_kernel: torch.Tensor = trainer.dsets[Datasplit.TRAIN].corruption.get_kernel( # pyright: ignore[reportOptionalMemberAccess]
                        conditioning=batch.corrupt_conditioning
                    )
                else:
                    raise ValueError("Cannot extract true kernel...")
                pred_kernel = physics.filter
                pred_kernel, true_kernel = equate_kernel_shapes(pred_kernel, true_kernel)
                pred_kernel_on_device = pred_kernel.to(true_kernel.device)
                vl["val_loss/kernel-mae"] = torch.mean(torch.abs(true_kernel - pred_kernel_on_device))
                # Multi-channel kernels are aligned globally on the anchor
                # channel, so inter-channel offsets still count.
                _knet = getattr(self.kernel_nn, "physics_net", self.kernel_nn)
                _anchor = (
                    getattr(_knet, "auto_center_anchor_channel", 0)
                    if pred_kernel_on_device.shape[-3] > 1 else None
                )
                vl["val_loss/kernel-mae-aligned"] = shift_invariant_kernel_mae(
                    pred_kernel_on_device, true_kernel, anchor_channel=_anchor,
                )
        except ValueError as e:
            warnings.warn(f"Cannot compute validation loss on kernels. Reason: {e}")
        return vl
