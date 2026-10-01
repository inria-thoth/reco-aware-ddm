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
from ddm4ip.utils.torch_utils import (
    center,
    equate_kernel_shapes,
    grad_norm,
    shift_invariant_kernel_mae,
)
from ddm4ip.utils.phema import FixedBetaEMA


class DiffInstructOnY(AbstractLoss):
    """Measurement-only objective F: distribution matching between simulated
    measurements A(x) + noise and the teacher trained on real measurements."""

    def __init__(
        self,
        config: Mapping[str, Any],
        aux_flow_nn: nn.Module,
        prtr_flow_nn: nn.Module,
        kernel_nn: DegradationNetwork,
    ):
        super().__init__(has_val_loss=True, n_accum_steps=config['loss']['n_accum_steps'])

        self.opt_state = OptimizationState(0, OptState.INNER)
        self.kernel_opt, self.lr_scheduler = create_optimizer(
            config['optim']['kernel'], kernel_nn.parameters()
        )
        gs_enabled = getattr(kernel_nn, "use_fp16", False)
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=gs_enabled)

        # torch.compile the two flow networks in place (identity is preserved for
        # FlowMatchingLoss / DDP / EMA). The recompile limit is raised above the
        # number of distinct UNet block shapes so that no block falls back to eager.
        if bool(config['loss'].get('compile', False)):
            import torch._dynamo as _dynamo
            for _attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(_dynamo.config, _attr):
                    setattr(_dynamo.config, _attr, 64)
            aux_flow_nn.compile(dynamic=True)
            prtr_flow_nn.compile(dynamic=True)

        # The auxiliary network is trained on the images generated with `kernel_nn`.
        config['loss']['aux']['n_accum_steps'] = self.n_accum_steps
        aux_cfg = dict(loss=config['loss']['aux'], optim=config['optim']['aux'])
        self.aux_flow_loss = FlowMatchingLoss(
            config=aux_cfg, flow_nn=aux_flow_nn, clean_images=True
        )
        self.aux_flow_nn = aux_flow_nn
        self.kernel_nn = kernel_nn
        self.kernel_nn_ddp = make_ddp(kernel_nn, broadcast_buffers=False)
        self.prtr_flow_nn = prtr_flow_nn.requires_grad_(False)

        # Fixed-decay EMA of the operator; validation and evaluation use its copy.
        self.kernel_ema_beta = None
        _beta = config['loss'].get('ema_kernel_beta', None)
        if _beta is not None:
            self.kernel_ema_beta = FixedBetaEMA(kernel_nn, beta=float(_beta))

    def get_images(self, start_img: torch.Tensor, return_physics=False, kernel_net=None, **kwargs):
        if kernel_net is None:
            kernel_net = self.kernel_nn_ddp
        return kernel_net(
            start_img,
            return_physics=return_physics,
            **kwargs
        )

    def loss_di(self, clean_img, conditioning, **kernel_kwargs):
        diff = self.aux_flow_loss.diffusion
        y1 = self.get_images(
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
            # Same dropout masks for the teacher and the auxiliary network.
            cuda_rng_state = torch.cuda.get_rng_state()
            prtr_velocity = self.prtr_flow_nn(yt, t, conditioning=conditioning)
            torch.cuda.set_rng_state(cuda_rng_state)
            aux_velocity = self.aux_flow_nn(yt, t, conditioning=conditioning)
            flow_diff = aux_velocity - prtr_velocity

            # Monitoring: flow-matching losses of the teacher and of the
            # auxiliary network on the simulated measurements; fm_diff -> 0 as
            # the simulated measurements match the real ones.
            fm_target = y1 - y0
            fm_prtr = ((prtr_velocity - fm_target) ** 2).mean()
            fm_aux = ((aux_velocity - fm_target) ** 2).mean()
            fm_diff = fm_prtr - fm_aux

        loss_di = (flow_diff * y1).mean()

        di_grad_norm = grad_norm(flow_diff)
        loss_info = {
           "loss/di": loss_di.detach(),
           "loss/di_grad_norm": di_grad_norm,
           "loss/fm_prtr": fm_prtr,
           "loss/fm_aux": fm_aux,
           "loss/fm_diff": fm_diff,
        }
        return loss_di, loss_info

    def __call__(self, trainer: BaseTrainer, batch: Batch):
        batch_dev = batch.cuda()
        assert batch_dev.clean is not None
        self.kernel_nn_ddp.train()

        if self.opt_state.state == OptState.OUTER:  # operator update
            with self.optimizer_step(self.kernel_opt, self.lr_scheduler, self.kernel_nn_ddp, grad_scaler=self.grad_scaler):
                loss_di, loss_info = self.loss_di(
                    batch_dev.clean,
                    batch_dev.clean_conditioning,
                    **batch_dev.noise_params,
                    **batch_dev.meta
                )
                loss_di = loss_di / self.n_accum_steps
                self.grad_scaler.scale(loss_di).backward()

            loss_info["learning_rate/kernel_nn"] = self.kernel_opt.param_groups[0]['lr']
            # In-place integer recentering of the operator parameter (no-op
            # unless enabled), before the EMA update so the EMA tracks it.
            self.kernel_nn.recenter_kernel_inplace_(self.kernel_opt)
            if self.kernel_ema_beta is not None:
                self.kernel_ema_beta.update()
            if super().is_full_step_complete():
                self.opt_state.increment_state()
        else:  # auxiliary-network update
            with torch.no_grad():
                self.kernel_nn.eval()
                y1 = self.get_images(
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
            aux_loss_info = self.aux_flow_loss(trainer, ker_batch)
            loss_info = {
                k.replace("loss/", "loss/aux_"): v for k, v in aux_loss_info.items()
            }
            if self.aux_flow_loss.is_full_step_complete():
                self.opt_state.increment_state()

        return loss_info

    def is_full_step_complete(self) -> bool:
        # A full step is one inner and one outer update.
        return super().is_full_step_complete() and self.opt_state.state == OptState.INNER

    def state_dict(self):
        return {
            "aux": self.aux_flow_loss.state_dict(),
            "kernel_opt": self.kernel_opt.state_dict(),
            "kernel_lr": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "gs": self.grad_scaler.state_dict(),
            "kernel_ema_beta": (
                self.kernel_ema_beta.state_dict()
                if self.kernel_ema_beta is not None else None
            ),
        }

    def load_state_dict(self, state_dict):
        self.aux_flow_loss.load_state_dict(state_dict["aux"])
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
        assert dev_batch.clean is not None
        kernel_for_val = self.kernel_ema_beta.ema if self.kernel_ema_beta is not None else None
        pred_corrupt, physics = self.get_images(
            dev_batch.clean,
            return_physics=True,
            kernel_net=kernel_for_val,
            conditioning=dev_batch.clean_conditioning,
            **dev_batch.noise_params,
            **dev_batch.meta
        )
        if dev_batch.corrupt is not None and dev_batch.corrupt.shape == pred_corrupt.shape:
            vl["val_loss/y-mae"] = torch.mean(torch.abs(dev_batch.corrupt - pred_corrupt))
        else:
            warnings.warn("Cannot compute validation loss on y.")

        try:
            if hasattr(physics, "filter") and physics.filter is not None:
                if dev_batch.kernel is not None:
                    true_kernel = dev_batch.kernel
                elif Datasplit.TRAIN in trainer.dsets and hasattr(trainer.dsets[Datasplit.TRAIN].corruption, 'get_kernel'): # pyright: ignore[reportOptionalMemberAccess]
                    true_kernel: torch.Tensor = trainer.dsets[Datasplit.TRAIN].corruption.get_kernel( # pyright: ignore[reportOptionalMemberAccess]
                        conditioning=batch.clean_conditioning
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
