import dataclasses
import enum
from typing import Any, Generic, TypeVar
import typing

import deepinv
import torch
import torch.utils.data

from ddm4ip.degradations.degradation import init_perturbation
from ddm4ip.utils import distributed


class Datasplit(enum.Enum):
    TRAIN = "train"
    TEST = "test"


@dataclasses.dataclass
class Batch:
    clean: torch.Tensor | None
    clean_label: torch.Tensor | None
    corrupt: torch.Tensor | None
    corrupt_label: torch.Tensor | None
    noise_params: dict[str, torch.Tensor]
    clean_conditioning: torch.Tensor | None
    corrupt_conditioning: torch.Tensor | None
    kernel: torch.Tensor | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def cuda(self):
        return self.to(device=torch.cuda.current_device())

    @property
    def batch_size(self):
        if self.clean is not None:
            return self.clean.shape[0]
        elif self.corrupt is not None:
            return self.corrupt.shape[0]
        raise RuntimeError("Cannot figure out batch size")

    def apply(self, fn):
        # At some point this should become recursive
        out = {}
        for k, v in vars(self).items():
            if isinstance(v, dict):
                out[k] = {}
                for k2, v2 in v.items():
                    out[k][k2] = fn(v2) if v2 is not None else None
            else:
                out[k] = fn(v) if v is not None else None
        return Batch(**out)

    def __getitem__(self, index):
        return self.apply(lambda var: var[index])

    def to(self, device):
        return self.apply(lambda var: var.to(device))

    @staticmethod
    def collate_fn(batch):
        def collate_one(lst):
            if lst[0] is None:
                assert all(el is None for el in lst)
                out = None
            else:
                out = torch.utils.data.default_collate(lst)
            return out
        return Batch(
            clean=collate_one([el.clean for el in batch]),
            clean_label=collate_one([el.clean_label for el in batch]),
            corrupt=collate_one([el.corrupt for el in batch]),
            corrupt_label=collate_one([el.corrupt_label for el in batch]),
            noise_params=torch.utils.data.default_collate([el.noise_params for el in batch]) if len(batch) > 0 else {},
            clean_conditioning=collate_one([el.clean_conditioning for el in batch]),
            corrupt_conditioning=collate_one([el.corrupt_conditioning for el in batch]),
            kernel=collate_one([el.kernel for el in batch]),
            meta=torch.utils.data.default_collate([el.meta for el in batch]) if len(batch) > 0 else {},
        )


def worker_init_fn(worker_id):
    from .data2 import ZipImagePath
    # Needed: the cache must be different per-process
    ZipImagePath.clear_cache()


CORR_TYPE = TypeVar('CORR_TYPE', deepinv.physics.Physics, None)

class DatasetType(typing.Protocol, Generic[CORR_TYPE]):
    corruption: CORR_TYPE
    corrupt_img_size: tuple[int, int, int] = (0, 0, 0)
    corrupt_conditioning_channels: int = 0
    clean_img_size: tuple[int, int, int] = (0, 0, 0)
    clean_conditioning_channels: int = 0
    label_dim: int

    def __getitem__(self, index) -> Batch:
        ...

    def __len__(self) -> int:
        ...

    @property
    def noise_params(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def init_dataloader(
    dset: torch.utils.data.Dataset[Batch],
    split: Datasplit,
    batch_size: int,
    num_workers: int,
    seed: int,
    start_idx: int,
    is_infinite: bool = True,
):
    from ddm4ip.utils.torch_utils import InfiniteSampler
    mp_ctx = None
    prefetch_factor = None
    if num_workers > 0:
        prefetch_factor = 2
    if split == Datasplit.TRAIN:
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
            collate_fn=Batch.collate_fn,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=mp_ctx,
            sampler=InfiniteSampler(
                dataset=dset,
                rank=distributed.get_rank(),
                num_replicas=distributed.get_world_size(),
                shuffle=True,
                seed=seed,
                start_idx=start_idx,
                is_infinite=is_infinite,
            )
        )
    else:
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
            collate_fn=Batch.collate_fn,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            sampler=InfiniteSampler(
                dataset=dset,
                rank=distributed.get_rank(),
                num_replicas=distributed.get_world_size(),
                shuffle=False,
                start_idx=start_idx,
                is_infinite=is_infinite,
            )
        )


def init_dataset(cfg: dict[str, Any], split: Datasplit, is_paired: bool):
    """Support initializing dataset from single configuration style:
    1. Specify train and test datasets in separate configurations
        under config['dataset']['train'] and config['dataset']['test'].
    """
    dset_cfg = cfg["dataset"]
    if split.value not in dset_cfg:
        raise RuntimeError(
            f"{split.name} dataset configuration has no path specified."
        )
    split_cfg = dset_cfg[split.value]
    device = split_cfg.get("device", "cuda")

    if "path" in split_cfg and split_cfg["path"] is not None:
        from ddm4ip.data.data2 import KnownCorruptionDataset
        return KnownCorruptionDataset(
            path=split_cfg["path"],
            corruption=init_perturbation(split_cfg["degradation"], split_cfg["noise"], device), # pyright: ignore[reportArgumentType]
            split=split,
            dset_cfg=split_cfg,
            shuffle_clean=split == Datasplit.TRAIN and not is_paired,
            device=device,
        )
    elif "clean_path" in split_cfg and "noisy_path" in split_cfg:
        from ddm4ip.data.data2 import UnpairedCorruptionDataset
        return UnpairedCorruptionDataset(
            clean_path=split_cfg["clean_path"],
            noisy_path=split_cfg["noisy_path"],
            corruption=init_perturbation(split_cfg.get("degradation", {}), split_cfg.get("noise", {}), device), # pyright: ignore[reportArgumentType]
            split=split,
            dset_cfg=split_cfg,
            shuffle_clean=split == Datasplit.TRAIN and not is_paired,
            device=device,
        )
    else:
        raise ValueError(f"Dataset for split {split.name} does not have correctly set paths.")
