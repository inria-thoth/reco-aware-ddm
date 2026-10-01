from typing import Any, Dict, Mapping

import torch

from ddm4ip.data.base import Batch
from ddm4ip.losses.base import AbstractLoss
from ddm4ip.losses.utils import create_optimizer, make_ddp
from ddm4ip.utils.torch_utils import center


class FlowMatchingDiffusion:
    """Rectified flow: x_t = (1 - t) x_0 + t x_1 with x_0 ~ N(0, I), t ~ U(0, 1),
    and velocity target x_1 - x_0."""

    @staticmethod
    def from_config(config: Mapping[str, Any]) -> 'FlowMatchingDiffusion':
        if config['loss']['diffusion']['type'] != "flowmatching":
            raise ValueError(f"Diffusion type {config['loss']['diffusion']['type']} not recognized.")
        return FlowMatchingDiffusion()

    def sample_noisy(self, clean_img: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(clean_img)  # x0

    def sample_time(self, batch_size: int, device: torch.device):
        return torch.rand([batch_size, 1, 1, 1], device=device)

    def make_xt(self, x_clean, x_noise, t):
        return (1 - t) * x_noise + t * x_clean

    def loss(self, net, x, t, noise, conditioning):
        xt = self.make_xt(x, noise, t)
        direction = net(xt, t.squeeze(), conditioning=conditioning)
        target_direction = x - noise
        loss = ((direction - target_direction) ** 2)
        return torch.mean(loss)


def flow_matching_loss(
    net: torch.nn.Module,
    diff: FlowMatchingDiffusion,
    images: torch.Tensor,
    conditioning: torch.Tensor | None = None,
):
    images = center(images)            # x1
    noisy = diff.sample_noisy(images)  # x0
    t = diff.sample_time(noisy.shape[0], noisy.device)
    loss = diff.loss(net, images, t, noisy, conditioning)
    loss_dict = {'flow_matching': loss.detach()}
    return loss, loss_dict


class FlowMatchingLoss(AbstractLoss):
    def __init__(self,
                 config: Mapping[str, Any],
                 flow_nn: torch.nn.Module,
                 clean_images: bool,):
        if hasattr(flow_nn, 'use_fp16'):
            fp16_flag = flow_nn.use_fp16
        else:
            fp16_flag = True

        super().__init__(
            has_val_loss=True,
            n_accum_steps=config['loss']['n_accum_steps'],
            gs_enabled=fp16_flag
        )
        self.opt, self.lr_scheduler = create_optimizer(config['optim'], flow_nn.parameters())
        self.flow_nn = flow_nn
        self.flow_nn_ddp = make_ddp(flow_nn)
        self.clean_images = clean_images
        self.diffusion = FlowMatchingDiffusion.from_config(config)

    def get_data_from_batch(self, batch: Batch):
        if self.clean_images:
            out_img, cond = batch.clean, batch.clean_conditioning
        else:
            out_img, cond = batch.corrupt, batch.corrupt_conditioning
        assert out_img is not None
        return out_img, cond

    def __call__(self, trainer, batch: Batch) -> Dict[str, torch.Tensor]:
        dev_batch = batch.cuda()
        self.flow_nn_ddp.train()

        target_img, cond = self.get_data_from_batch(dev_batch)
        with self.optimizer_step(self.opt, self.lr_scheduler, self.flow_nn_ddp, grad_scaler=self.grad_scaler):
            loss, loss_dict = flow_matching_loss(self.flow_nn_ddp, self.diffusion, target_img, cond)
            loss = loss / self.n_accum_steps
            self.grad_scaler.scale(loss).backward()
        return {
            "learning_rate/flow_nn": self.opt.param_groups[0]['lr'],
        } | {f"loss/{k}": v for k, v in loss_dict.items()}

    @torch.no_grad()
    def val_loss(self, trainer, batch: Batch):
        dev_batch = batch.cuda()
        self.flow_nn.eval()
        target_img, cond = self.get_data_from_batch(dev_batch)
        _, loss_dict = flow_matching_loss(self.flow_nn, self.diffusion, target_img, cond)
        return {
            f"val_loss/{k}": v for k, v in loss_dict.items()
        }

    def state_dict(self):
        return {
            "opt": self.opt.state_dict(),
            "lr": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "gs": self.grad_scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict["opt"])
        if self.lr_scheduler is not None:
            if "lr" not in state_dict:
                raise ValueError(
                    "Created loss with lr scheduler but state-dict has no lr scheduler state."
                )
            self.lr_scheduler.load_state_dict(state_dict["lr"])
        self.grad_scaler.load_state_dict(state_dict['gs'])
