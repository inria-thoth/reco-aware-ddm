import os
import abc
from typing import Any, Tuple, ClassVar, Sequence
from typing_extensions import override
from pathlib import Path
import functools
import zipfile
import dataclasses

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.v2 as v2
import deepinv

from ddm4ip.data.base import Batch, DatasetType, Datasplit
from ddm4ip.data.utils import (
    AddLocMapTransform,
    GetAllPatchesTransform,
    get_dim_blocks,
)
from ddm4ip.utils import distributed
from ddm4ip.utils.distributed import get_local_rank

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.pt',
]

def file_ext(fname):
    return os.path.splitext(fname)[1].lower()


def _load_rgb_any_depth(src) -> np.ndarray:
    """Read an image to an HWC, RGB, float32 [0,1] ndarray WITHOUT dropping bit
    depth.

    `src` is a path (str) or an encoded-bytes buffer (np.uint8 ndarray). cv2's
    IMREAD_UNCHANGED keeps 16-bit as uint16 (unlike PIL's convert("RGB"), which
    truncates to 8-bit). We normalise here by the NATIVE bit depth (255 for
    uint8, 65535 for uint16), so the result is correct [0,1] regardless of depth
    and independent of torchvision's integer-scaling support. The downstream
    ToImage + ToDtype(float32, scale=True) is then a no-op on this float array."""
    if isinstance(src, np.ndarray):
        img = cv2.imdecode(src, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image from {src!r}")
    if img.ndim == 2:                       # grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:                                   # drop any alpha, BGR -> RGB
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    maxv = 65535.0 if img.dtype == np.uint16 else 255.0
    return np.ascontiguousarray(img.astype(np.float32) / maxv)


class AbstractImagePath(abc.ABC):
    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def load_raw_img(self) -> Image.Image:
        pass

    @staticmethod
    @abc.abstractmethod
    def load_all_paths(root: Path, **kwargs) -> Sequence['AbstractImagePath']:
        pass


@dataclasses.dataclass
@functools.total_ordering
class ImagePath(AbstractImagePath):
    root: Path
    name: str

    @override
    def __lt__(self, other: 'ImagePath'):
        return (self.name, self.root) < (other.name, other.root)

    @override
    def load_raw_img(self) -> np.ndarray:
        # HWC RGB ndarray, native bit depth preserved (uint8 or uint16). PIL's
        # convert("RGB") would truncate 16-bit -> 8-bit; cv2 IMREAD_UNCHANGED
        # keeps it, and the downstream ToImage + ToDtype(scale=True) rescales by
        # the dtype max (255 or 65535) to [0,1].
        return _load_rgb_any_depth(str(self.root / self.name))

    @override
    @staticmethod
    def load_all_paths(root: Path, **kwargs):
        fnames = [
            str(f.relative_to(root))
            for f in Path(root).rglob("*")
            if f.suffix in torchvision.datasets.folder.IMG_EXTENSIONS
        ]
        fnames = sorted(fnames)
        return [ImagePath(root, fname) for fname in fnames]

    def __hash__(self) -> int:
        return hash((self.root, self.name))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ImagePath):
            return self.root == value.root and self.name == value.name
        return False


@dataclasses.dataclass
@functools.total_ordering
class ZipImagePath(AbstractImagePath):
    zip_cache: ClassVar[dict[str, zipfile.ZipFile]] = dict()

    file: str | Path
    name: str

    @override
    def __lt__(self, other: 'ZipImagePath'):
        return (self.name, self.file) < (other.name, other.file)

    @override
    def load_raw_img(self) -> np.ndarray:
        # Depth-preserving read from the zip (see ImagePath.load_raw_img).
        zfile = self.get_zipfile(self.file)
        with zfile.open(self.name, 'r') as f:
            data = f.read()
        return _load_rgb_any_depth(np.frombuffer(data, np.uint8))

    @classmethod
    def get_zipfile(cls, path: str | Path) -> zipfile.ZipFile:
        if (file := cls.zip_cache.get(str(path))) is None:
            file = zipfile.ZipFile(str(path))
            cls.zip_cache[str(path)] = file
        return file

    @classmethod
    def clear_cache(cls):
        cls.zip_cache = dict()

    @override
    @staticmethod
    def load_all_paths(root: Path, **kwargs):
        try:
            with zipfile.ZipFile(str(root)) as zfile:
                names = sorted(
                    filter(
                        lambda n: os.path.splitext(n)[1] in torchvision.datasets.folder.IMG_EXTENSIONS,
                        zfile.namelist()
                    )
                )
                return [ZipImagePath(root, f) for f in names]
        except zipfile.BadZipFile as e:
            raise zipfile.BadZipFile(f"Error when opening zip at {str(root)}: {e}") from e

    def __hash__(self) -> int:
        return hash((self.file, self.name))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ZipImagePath):
            return self.file == value.file and self.name == value.name
        return False


def load_all_paths(root: Path):
    if root.suffix == ".zip" and root.is_file():
        return ZipImagePath.load_all_paths(root)
    elif root.is_dir():
        all_files = ImagePath.load_all_paths(root)
        # Load from zips
        for zip_file in root.rglob("*.zip"):
            all_files.extend(ZipImagePath.load_all_paths(zip_file)) # pyright: ignore[reportArgumentType]
        return all_files
    elif file_ext(root) in IMG_EXTENSIONS:
           return [ImagePath(root.parent, root.name)]
    else:
        raise ValueError(f"{root} not recognized as image dataset root.")


def search_idx_in_list(all_num_patches: list[int], idx: int) -> tuple[int, int]:
    cs = np.cumsum(all_num_patches)
    full_img_idx: int = np.searchsorted(cs, idx, side="right") # type: ignore
    if full_img_idx == 0:
        remainder = idx
    elif full_img_idx == len(cs):
        raise IndexError(f"Index {idx} invalid.")
    else:
        remainder = idx - cs[full_img_idx - 1]
    return full_img_idx, remainder


class ImageFolderDataset(torch.utils.data.Dataset[torch.Tensor]):
    """Base dataset loading images from a folder or zip file"""
    def __init__(
        self,
        base_path: Path | str,
        transforms: v2.Transform | None,
        shuffle: bool,
    ):
        self.transforms = transforms if transforms is not None else v2.Identity()
        if isinstance(base_path, str):
            base_path = os.path.expanduser(base_path)
        self.root = Path(base_path)
        self.files = load_all_paths(self.root)
        if shuffle:
            self.files = [self.files[i] for i in torch.randperm(len(self.files))]

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        return self.transforms(path.load_raw_img())

    def __len__(self) -> int:
        return len(self.files)


class BaseDataset(torch.utils.data.Dataset[Batch], DatasetType, abc.ABC):
    def __init__(self, split: Datasplit, dset_cfg: dict[str, Any], shuffle_clean: bool = True):
        self.split = split
        self.shuffle_clean = shuffle_clean
        self.patch_size: int | None = dset_cfg.get("patch_size", None)
        # Optional crop size of the degraded stream (defaults to `patch_size`). In
        # blind super-resolution the observed set is made of undegraded crops of
        # the low-resolution image at the post-decimation size.
        self.noisy_patch_size: int | None = dset_cfg.get("noisy_patch_size", None)
        self.label_dim = 0  # NOTE: always fixed, should be removed.

        # Parse conditioning parameters
        self.space_conditioning: bool = dset_cfg.get("space_conditioning", False)

        self.need_clean: bool = dset_cfg.get("need_clean", True)
        self.need_noisy: bool = dset_cfg.get("need_noisy", True)
        assert self.need_clean or self.need_noisy

        self.num_patches_per_image = 1
        if self.patch_size is not None and self.split == Datasplit.TRAIN:
            self.num_patches_per_image = dset_cfg.get("num_patches_per_image", 1)

        self.init_base_dset()
        self.dset_length = self.init_length()

        # Set-up necessary attributes by running pipeline
        batch = self[0]
        self.clean_img_size = (0, 0, 0)
        if batch.clean is not None:
            self.clean_img_size = (batch.clean.shape[-3], batch.clean.shape[-2], batch.clean.shape[-1])
        self.clean_conditioning_channels = 0
        if batch.clean_conditioning is not None:
            self.clean_conditioning_channels = batch.clean_conditioning.shape[-3]
        self.corrupt_img_size = (0, 0, 0)
        if batch.corrupt is not None:
            self.corrupt_img_size = (batch.corrupt.shape[-3], batch.corrupt.shape[-2], batch.corrupt.shape[-1])
        self.corrupt_conditioning_channels = 0
        if batch.corrupt_conditioning is not None:
            self.corrupt_conditioning_channels = batch.corrupt_conditioning.shape[-3]

    @abc.abstractmethod
    def init_base_dset(self):
        pass

    @abc.abstractmethod
    def init_length(self) -> int:
        pass

    def get_num_patches(self, img: torch.Tensor):
        if self.patch_size is None:
            return 1
        h, w = img.shape[-2], img.shape[-1]
        num_img_patches_h = get_dim_blocks(h, self.patch_size, stride=self.patch_size)
        num_img_patches_w = get_dim_blocks(w, self.patch_size, stride=self.patch_size)
        return num_img_patches_h * num_img_patches_w

    def split_img_cond(self, full: torch.Tensor | None) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        img, cond = None, None
        if self.space_conditioning and full is not None:
            cond = full[..., -2:, :, :]
            img = full[..., :-2, :, :]
        elif full is not None:
            img = full
        return img, cond

    def add_cond(self, img: torch.Tensor, conditioning: torch.Tensor | None) -> torch.Tensor:
        if conditioning is None:
            return img
        if img.shape[-1] == conditioning.shape[-1] and img.shape[-2] == conditioning.shape[-2]:
            return torch.cat((img, conditioning), dim=-3)
        raise RuntimeError(f"Cannot add conditioning of shape {conditioning.shape} to image of shape {img.shape}")

    def corrupt_patch(self, full_patch: torch.Tensor, corruption: deepinv.physics.Physics | None) -> torch.Tensor:
        if corruption is None:
            return full_patch
        patch, cond = self.split_img_cond(full_patch)
        assert patch is not None
        squeeze_out = False
        if patch.dim() == 3:
            squeeze_out = True
            patch = patch[None, ...]
        if cond is not None and cond.dim() == 3:
            cond = cond[None, ...]
        corrupted = corruption(patch, conditioning=cond)
        full_patch = self.add_cond(corrupted, cond)
        if squeeze_out:
            return full_patch.squeeze(0)
        return full_patch

    @functools.cache
    def get_train_transform(self, img_size: tuple[int, int],
                            patch_size: int | None = None) -> v2.Transform:
        """`patch_size` overrides self.patch_size (used for the degraded stream)."""
        patch_size = self.patch_size if patch_size is None else patch_size
        trsf: list[v2.Transform] = [v2.Identity()]
        if patch_size is not None:
            trsf.append(v2.RandomCrop(patch_size, pad_if_needed=True))
        return v2.Compose(trsf)

    @abc.abstractmethod
    def getitem_train(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pass

    @functools.cache
    def get_test_transform(self, img_size: tuple[int, int]) -> v2.Transform:
        trsf: list[v2.Transform] = []
        if self.patch_size is not None:
            trsf.append(GetAllPatchesTransform(self.patch_size, stride=self.patch_size))
        else:
            # transform must add a leading batch dimension to work with downstream processing
            trsf.append(v2.Lambda(lambda img: img[None, ...]))
        return v2.Compose(trsf)

    @abc.abstractmethod
    def getitem_test(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pass

    def __getitem__(self, index: int) -> Batch:
        if self.split == Datasplit.TEST:
            clean_full, noisy_full = self.getitem_test(index)
        else:
            clean_full, noisy_full = self.getitem_train(index)

        clean_patch, clean_cond = self.split_img_cond(clean_full)
        noisy_patch, noisy_cond = self.split_img_cond(noisy_full)
        return Batch(
            clean=clean_patch,
            corrupt=noisy_patch,
            clean_label=None,
            corrupt_label=None,
            noise_params=self.noise_params,
            clean_conditioning=clean_cond,
            corrupt_conditioning=noisy_cond,
        )

    @property
    @abc.abstractmethod
    def noise_params(self) -> dict[str, torch.Tensor]:
        pass

    def __len__(self):
        return self.dset_length

class ToDevice:
    def __init__(self, device):
        self.device = device
    def __call__(self, tensor):
        return tensor.to(self.device)

class KnownCorruptionDataset(BaseDataset):
    def __init__(
        self,
        path: str | Path,
        corruption: deepinv.physics.Physics,
        split: Datasplit,
        dset_cfg: dict[str, Any],
        device: torch.device,
        shuffle_clean: bool = True,
    ):
        self.path = Path(path)
        # Will be initialized
        self.dset = None
        self.clean_cache, self.noisy_cache = None, None
        self.clean_full_ids = None
        self.tst_n_patches = None
        if device == "cuda":
            self.device = torch.device("cuda", get_local_rank())
        else:
            self.device = torch.device("cpu")        
        self.corruption: deepinv.physics.Physics | None = corruption
        if self.corruption is not None:
            self.corruption = self.corruption.to(device)
        super().__init__(split, dset_cfg, shuffle_clean)
        assert self.dset is not None

        distributed.print0(f"Loaded KnownCorruptionDataset dataset at '{path}':")
        distributed.print0(f"Split:                          {self.split}")
        distributed.print0(f"Clean patch size:               {self.clean_img_size}")
        distributed.print0(f"Noisy patch size:               {self.corrupt_img_size}")
        distributed.print0(f"Clean conditioning channels:    {self.clean_conditioning_channels}")
        distributed.print0(f"Noisy conditioning channels:    {self.corrupt_conditioning_channels}")
        distributed.print0(f"Space conditioning:             {self.space_conditioning}")
        distributed.print0(f"Number of base images:          {len(self.dset)}")
        if self.split == Datasplit.TEST:
            distributed.print0(f"Dataset length:                 {self.dset_length}")
        else:
            distributed.print0(f"Number patches per image:       {self.num_patches_per_image}")
        distributed.print0()

    @override
    def init_base_dset(self):
        img_trsf = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if self.space_conditioning:
            img_trsf.append(AddLocMapTransform())
        img_trsf.append(ToDevice(self.device))
        self.dset = ImageFolderDataset(
            base_path=self.path, transforms=v2.Compose(img_trsf), shuffle=False
        )

        # Initialize separate indices into the full dataset for clean
        if self.split == Datasplit.TRAIN and self.need_clean:
            if self.shuffle_clean:
                self.clean_full_ids = torch.randperm(len(self.dset)).tolist()
            else:
                self.clean_full_ids = torch.arange(len(self.dset)).tolist()

    @override
    def init_length(self):
        assert self.dset is not None
        if self.split == Datasplit.TRAIN:
            return self.num_patches_per_image * len(self.dset)
        else:
            self.tst_n_patches = [self.get_num_patches(full_img) for full_img in self.dset]
            return sum(self.tst_n_patches)

    @override
    def getitem_train(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.dset is not None
        clean_full, noisy_full = None, None
        if self.need_clean:  # get clean patch with conditioning
            assert self.clean_full_ids is not None
            if self.clean_cache is None or len(self.clean_cache) == 0:
                self.clean_cache = []
                img = self.dset[self.clean_full_ids[index // self.num_patches_per_image]]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]))
                for _ in range(self.num_patches_per_image):
                    self.clean_cache.append(trsf(img))
            assert isinstance(self.clean_cache, list)
            clean_full = self.clean_cache.pop()
        if self.need_noisy:  # get noisy patch with conditioning
            if self.noisy_cache is None or len(self.noisy_cache) == 0:
                self.noisy_cache = []
                img = self.dset[index // self.num_patches_per_image]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]),
                                                patch_size=self.noisy_patch_size)
                for _ in range(self.num_patches_per_image):
                    # The degradation is applied here.
                    self.noisy_cache.append(self.corrupt_patch(trsf(img), self.corruption))
            assert isinstance(self.noisy_cache, list)
            noisy_full = self.noisy_cache.pop()
        return clean_full, noisy_full

    @override
    def getitem_test(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.tst_n_patches is not None
        assert self.dset is not None
        clean_full, noisy_full = None, None
        img_idx, patch_idx = search_idx_in_list(self.tst_n_patches, idx=index)
        # We fetch image only once and always put in the clean cache
        if self.clean_cache is None or self.clean_cache[0] != img_idx:
            self.clean_cache = (img_idx, [])
            img = self.dset[img_idx]
            trsf = self.get_test_transform((img.shape[-2], img.shape[-1]))
            transformed_image = trsf(img)
            if len(transformed_image.shape) == 4:
                for patch in trsf(img).unbind(0):
                    self.clean_cache[1].append(patch)
            else:
                self.clean_cache[1].append(transformed_image)

        if self.need_clean:
            clean_full = self.clean_cache[1][patch_idx]
        if self.need_noisy:
            # The degradation is applied here.
            noisy_full = self.corrupt_patch(self.clean_cache[1][patch_idx], self.corruption)

        return clean_full, noisy_full

    @property
    def noise_params(self) -> dict[str, torch.Tensor]:
        try:
            return dict(self.corruption.noise_model.named_buffers())
        except AttributeError:
            return {}


class UnpairedCorruptionDataset(BaseDataset):
    def __init__(
        self,
        clean_path: str | Path,
        noisy_path: str | Path,
        corruption: deepinv.physics.Physics | None,
        split: Datasplit,
        dset_cfg: dict[str, Any],
        shuffle_clean: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.clean_path = Path(clean_path)
        self.noisy_path = Path(noisy_path)
        self.clean_dset, self.noisy_dset = None, None
        self.clean_cache, self.noisy_cache = None, None
        self.tst_n_patches = None
        if device == "cuda":
            self.device = torch.device("cuda", get_local_rank())
        else:
            self.device = torch.device("cpu")        
        self.corruption: deepinv.physics.Physics | None = corruption
        if self.corruption is not None:
            self.corruption = self.corruption.to(device)
        super().__init__(split, dset_cfg, shuffle_clean)

        distributed.print0(f"Loaded UnpairedCorruptionDataset dataset at '{self.clean_path}' and '{self.noisy_path}':")
        distributed.print0(f"Split:                          {self.split}")
        distributed.print0(f"Clean patch size:               {self.clean_img_size}")
        distributed.print0(f"Noisy patch size:               {self.corrupt_img_size}")
        distributed.print0(f"Clean conditioning channels:    {self.clean_conditioning_channels}")
        distributed.print0(f"Noisy conditioning channels:    {self.corrupt_conditioning_channels}")
        distributed.print0(f"Space conditioning:             {self.space_conditioning}")
        distributed.print0(f"Number of base clean images:    {len(self.clean_dset) if self.clean_dset is not None else 0}")
        distributed.print0(f"Number of base corrupt images:  {len(self.noisy_dset) if self.noisy_dset is not None else 0}")
        if self.split == Datasplit.TEST:
            distributed.print0(f"Dataset length:                 {self.dset_length}")
        else:
            distributed.print0(f"Number patches per image:       {self.num_patches_per_image}")
        distributed.print0()

    @override
    def init_base_dset(self):
        if self.need_clean:
            img_trsf = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            if self.space_conditioning:
                img_trsf.append(AddLocMapTransform())
            img_trsf.append(ToDevice(self.device))
            self.clean_dset = ImageFolderDataset(
                base_path=self.clean_path, transforms=v2.Compose(img_trsf), shuffle=self.shuffle_clean
            )
        if self.need_noisy:
            img_trsf = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            if self.space_conditioning:
                img_trsf.append(AddLocMapTransform())
            img_trsf.append(ToDevice(self.device))
            self.noisy_dset = ImageFolderDataset(
                base_path=self.noisy_path, transforms=v2.Compose(img_trsf), shuffle=False
            )
        if (
            self.split == Datasplit.TEST and
            self.clean_dset is not None and self.noisy_dset is not None and
            len(self.clean_dset) != len(self.noisy_dset)
        ):
            raise RuntimeError("Test datasets cannot have different lengths as they are paired.")

    @override
    def init_length(self):
        if self.split == Datasplit.TRAIN:
            return self.num_patches_per_image * max(
                len(self.clean_dset) if self.clean_dset is not None else 0,
                len(self.noisy_dset) if self.noisy_dset is not None else 0
            )
        else:
            tst_dset = self.clean_dset or self.noisy_dset
            assert tst_dset is not None
            self.tst_n_patches = [self.get_num_patches(full_img) for full_img in tst_dset]
            return sum(self.tst_n_patches)

    @override
    def getitem_train(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        clean_full, noisy_full = None, None
        if self.need_clean:  # get clean patch with conditioning
            assert self.clean_dset is not None
            if self.clean_cache is None or len(self.clean_cache) == 0:
                self.clean_cache = []
                img = self.clean_dset[(index // self.num_patches_per_image) % len(self.clean_dset)]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]))
                for _ in range(self.num_patches_per_image):
                    self.clean_cache.append(trsf(img))
            assert isinstance(self.clean_cache, list)
            clean_full = self.clean_cache.pop()
        if self.need_noisy:  # get noisy patch with conditioning
            assert self.noisy_dset is not None
            if self.noisy_cache is None or len(self.noisy_cache) == 0:
                self.noisy_cache = []
                img = self.noisy_dset[(index // self.num_patches_per_image) % len(self.noisy_dset)]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]),
                                                patch_size=self.noisy_patch_size)
                for _ in range(self.num_patches_per_image):
                    # The degradation is applied here.
                    self.noisy_cache.append(self.corrupt_patch(trsf(img), self.corruption))
            assert isinstance(self.noisy_cache, list)
            noisy_full = self.noisy_cache.pop()

        return clean_full, noisy_full

    @override
    def getitem_test(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.tst_n_patches is not None
        clean_full, noisy_full = None, None
        img_idx, patch_idx = search_idx_in_list(self.tst_n_patches, idx=index)
        if self.need_clean:
            assert self.clean_dset is not None
            if self.clean_cache is None or self.clean_cache[0] != img_idx:
                self.clean_cache = (img_idx, [])
                img = self.clean_dset[img_idx]
                trsf = self.get_test_transform((img.shape[-2], img.shape[-1]))
                for patch in trsf(img).unbind(0):
                    self.clean_cache[1].append(patch)
            clean_full = self.clean_cache[1][patch_idx]
        if self.need_noisy:
            assert self.noisy_dset is not None
            if self.noisy_cache is None or self.noisy_cache[0] != img_idx:
                self.noisy_cache = (img_idx, [])
                img = self.noisy_dset[img_idx]
                trsf = self.get_test_transform((img.shape[-2], img.shape[-1]))
                for patch in trsf(img).unbind(0):
                    self.noisy_cache[1].append(patch)
            # The degradation is applied here.
            noisy_full = self.corrupt_patch(self.noisy_cache[1][patch_idx], self.corruption)
        return clean_full, noisy_full

    @property
    def noise_params(self) -> dict[str, torch.Tensor]:
        if self.corruption is None:
            return {}
        try:
            return dict(self.corruption.noise_model.named_buffers())
        except AttributeError:
            return {}
