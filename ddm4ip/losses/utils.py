import itertools

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from ddm4ip.utils import distributed


def create_optimizer(config, *parameters):
    """AdamW (betas 0.9/0.999, weight decay 0.01) with the learning rate `lr`, and
    optionally (`lr_scheduler.type: CosineAnnealingLR`) a linear warm-up over
    `rampup_nimg` images followed by a cosine decay to `eta_min` over `max_nimg`."""
    concat_params = itertools.chain(*parameters)
    optim = torch.optim.AdamW(
        concat_params,
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    lr_scheduler = None
    if config.get('lr_scheduler') is not None:
        if config['lr_scheduler']['type'] != "CosineAnnealingLR":
            raise ValueError(f"lr_scheduler type '{config['lr_scheduler']['type']}' is not supported.")
        iter_batch_size = config['lr_scheduler']['batch_size'] * distributed.get_world_size()
        warmup_iter = config['lr_scheduler']['rampup_nimg'] // iter_batch_size
        warmup_lr_sched = LinearLR(optim, start_factor=1/1000, end_factor=1, total_iters=warmup_iter)
        cosine_iter_max = config['lr_scheduler']['max_nimg'] // iter_batch_size
        cosine_lr_sched = CosineAnnealingLR(optim, T_max=cosine_iter_max,  eta_min=config['lr_scheduler'].get('eta_min', 1e-7))
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim,
            [warmup_lr_sched, cosine_lr_sched],
            [warmup_iter]
        )
    return optim, lr_scheduler


def make_ddp(model: torch.nn.Module, **kwargs):
    if distributed.get_world_size() > 1:
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[distributed.get_local_rank()],
            **kwargs
        )
    return model
