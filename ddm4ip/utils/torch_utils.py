from collections.abc import Callable
from io import BytesIO
import math
from pathlib import Path
from typing import Sequence, Tuple, TypeVar
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.v2.functional as TF
import cv2


FT = TypeVar("FT", float, torch.Tensor)

def center(img: FT, max_val: float = 1.0) -> FT:
    # [0, max_val] -> [-max_val, max_val]
    return img * 2 - max_val


def shift_invariant_kernel_mae(
    pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-6,
    bg_quantile: float = 0.0, anchor_channel: int | None = None,
) -> torch.Tensor:
    """MAE between predicted and ground-truth kernels after COM alignment.

    Blind deconvolution is identifiable only up to a translation
    (K_tau * x_{-tau} = K * x). The standard pointwise MAE between a
    learned kernel and the ground truth is sensitive to the residual
    shift even when the kernel *shape* has been recovered correctly.
    This metric shifts the prediction so its centre of mass matches the
    ground-truth's COM (sub-pixel, via bilinear ``grid_sample``), then
    reports the standard MAE. Useful as a "is the shape right" diagnostic
    alongside the raw MAE.

    ``bg_quantile`` subtracts a per-kernel background floor before the COM
    (see ``kernel_com``); leave at 0 for the historical plain-``|K|`` COM.
    Note this only changes the *alignment*, not the reported MAE values.

    ``anchor_channel``:
      * ``None`` (default): the alignment shift is computed per leading
        "kernel" (last two dims spatial, everything else — INCLUDING the
        channel dim — treated as batch). For a multi-channel kernel this
        aligns each channel INDEPENDENTLY, which removes any genuine
        inter-channel displacement (e.g. chromatic aberration) and so
        over-states the recovery quality.
      * an int: treat the dim before the spatial dims as channels, compute
        ONE alignment shift from that channel of pred vs true, and apply it
        GLOBALLY to all channels (single grid per kernel). This preserves
        inter-channel offsets — consistent with green-anchored in-place
        recentring (``auto_center_anchor_channel``). Use this for per-RGB-
        channel / chromatic-aberration kernels so the aligned MAE still
        penalises a wrong relative channel displacement.
    """
    pred, true = pred.detach(), true.detach()
    H, W = pred.shape[-2], pred.shape[-1]

    if anchor_channel is None:
        # Per-leading-kernel alignment (channel folded into batch).
        orig_shape = pred.shape
        pred_flat = pred.reshape(-1, 1, H, W)
        true_flat = true.reshape(-1, 1, H, W)
        B = pred_flat.shape[0]
        com_p_y, com_p_x = kernel_com(pred_flat, eps=eps, bg_quantile=bg_quantile)
        com_t_y, com_t_x = kernel_com(true_flat, eps=eps, bg_quantile=bg_quantile)
        dy = (com_t_y - com_p_y).view(B)
        dx = (com_t_x - com_p_x).view(B)
    else:
        # Global (anchor-channel) alignment: one shift per kernel, shared
        # across channels. Collapse all dims before the channel into batch.
        C = pred.shape[-3]
        orig_shape = pred.shape
        pred_flat = pred.reshape(-1, C, H, W)
        true_flat = true.reshape(-1, C, H, W)
        B = pred_flat.shape[0]
        a = max(0, min(int(anchor_channel), C - 1))
        com_p_y, com_p_x = kernel_com(
            pred_flat[:, a:a + 1], eps=eps, bg_quantile=bg_quantile)
        com_t_y, com_t_x = kernel_com(
            true_flat[:, a:a + 1], eps=eps, bg_quantile=bg_quantile)
        dy = (com_t_y - com_p_y).view(B)
        dx = (com_t_x - com_p_x).view(B)

    # Build a sampling grid offset by -(dx, dy) in normalised coords. With
    # input (B, c, H, W) and grid (B, H, W, 2), grid_sample applies the SAME
    # grid to every channel of each batch item — so the anchor branch shifts
    # all channels by the single anchor shift.
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=pred.device, dtype=pred.dtype),
        torch.linspace(-1, 1, W, device=pred.device, dtype=pred.dtype),
        indexing='ij',
    )
    gy = gy.unsqueeze(0).expand(B, -1, -1).clone()
    gx = gx.unsqueeze(0).expand(B, -1, -1).clone()
    gy = gy - (dy * 2.0 / max(H - 1, 1)).view(B, 1, 1)
    gx = gx - (dx * 2.0 / max(W - 1, 1)).view(B, 1, 1)
    grid = torch.stack([gx, gy], dim=-1)  # (B, H, W, 2), last dim is (x, y)
    pred_aligned = torch.nn.functional.grid_sample(
        pred_flat, grid, mode='bilinear', padding_mode='zeros',
        align_corners=True,
    )
    pred_aligned = pred_aligned.reshape(orig_shape)
    true = true.reshape(orig_shape)
    return torch.mean(torch.abs(true - pred_aligned))


def kernel_com(
    K_flat: torch.Tensor, eps: float = 1e-8, bg_quantile: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-kernel centre of mass (in pixel-index coords) of ``|K|``.

    ``K_flat`` is ``(B, C, H, W)``; returns ``(com_y, com_x)`` of shape
    ``(B, C, 1, 1)``. ``bg_quantile`` > 0 subtracts that quantile of ``|K|`` (per
    kernel) and clamps at 0 before computing the centre of mass, so that a
    uniform pedestal (e.g. from a softplus parameterisation) does not pull the
    estimate toward the geometric centre.
    """
    H, W = K_flat.shape[-2], K_flat.shape[-1]
    Ka = _remove_background(K_flat.abs(), bg_quantile)
    ys = torch.arange(H, device=K_flat.device, dtype=K_flat.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=K_flat.device, dtype=K_flat.dtype).view(1, 1, 1, W)
    Ka_sum = Ka.sum(dim=(-1, -2), keepdim=True) + eps
    com_y = (Ka * ys).sum(dim=(-1, -2), keepdim=True) / Ka_sum
    com_x = (Ka * xs).sum(dim=(-1, -2), keepdim=True) / Ka_sum
    return com_y, com_x


def _remove_background(Ka: torch.Tensor, bg_quantile: float) -> torch.Tensor:
    if bg_quantile > 0:
        # Per-(batch, channel) floor, reshaped to (B, C, 1, 1).
        bg = torch.quantile(
            Ka.flatten(start_dim=-2), q=bg_quantile, dim=-1
        ).reshape(*Ka.shape[:-2], 1, 1)
        Ka = (Ka - bg).clamp_min(0)
    return Ka


def windowed_kernel_com(
    K_flat: torch.Tensor,
    window: float,
    n_iter: int = 5,
    eps: float = 1e-8,
    bg_quantile: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-shift centre of mass: the centroid of the dominant blob of ``|K|``.

    Starts from the global (background-subtracted) centre of mass, then
    repeatedly recomputes it using only the mass within ``window`` pixels
    (L-inf box) of the current estimate, so that distant outlier mass cannot
    move it. ``K_flat`` is ``(B, C, H, W)``; returns ``(com_y, com_x)`` of shape
    ``(B, C, 1, 1)``.
    """
    H, W = K_flat.shape[-2], K_flat.shape[-1]
    Ka = _remove_background(K_flat.abs(), bg_quantile)
    ys = torch.arange(H, device=K_flat.device, dtype=K_flat.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=K_flat.device, dtype=K_flat.dtype).view(1, 1, 1, W)

    s = Ka.sum(dim=(-1, -2), keepdim=True) + eps
    cy = (Ka * ys).sum(dim=(-1, -2), keepdim=True) / s
    cx = (Ka * xs).sum(dim=(-1, -2), keepdim=True) / s

    for _ in range(max(1, n_iter)):
        mask = (
            (ys - cy).abs() <= window
        ) & (
            (xs - cx).abs() <= window
        )
        Kw = Ka * mask.to(Ka.dtype)
        sw = Kw.sum(dim=(-1, -2), keepdim=True) + eps
        cy = (Kw * ys).sum(dim=(-1, -2), keepdim=True) / sw
        cx = (Kw * xs).sum(dim=(-1, -2), keepdim=True) / sw
    return cy, cx


def center_kernel_via_shift(
    kernel: torch.Tensor, eps: float = 1e-8, bg_quantile: float = 0.5,
    window: float = 8.0, n_iter: int = 5, anchor_channel: int | None = None,
) -> torch.Tensor:
    """Shift ``kernel`` so that its windowed centre of mass (see
    ``windowed_kernel_com``) lies at the geometric centre, with a differentiable
    bilinear ``grid_sample``. The centre of mass is computed without gradient,
    so the gradient only flows through the interpolation.

    With ``anchor_channel=None`` each leading kernel (channels included) is
    centred independently; with an int, one shift is computed on that channel
    and applied to all channels, which preserves inter-channel offsets.

    Accepts ``(..., H, W)``. Mass shifted outside the frame is lost (zero
    padding), so renormalise afterwards if needed.
    """
    orig_shape = kernel.shape
    H, W = kernel.shape[-2], kernel.shape[-1]

    if anchor_channel is None:
        K_grid = kernel.reshape(-1, 1, H, W)   # what grid_sample shifts
        K_com = K_grid                         # what the centre of mass is measured on
    else:
        C = kernel.shape[-3]
        a = max(0, min(int(anchor_channel), C - 1))
        K_grid = kernel.reshape(-1, C, H, W)
        K_com = K_grid[:, a:a + 1]
    B = K_grid.shape[0]

    with torch.no_grad():
        com_y, com_x = windowed_kernel_com(
            K_com, window=window, n_iter=n_iter,
            eps=eps, bg_quantile=bg_quantile,
        )
        target_y = (H - 1) / 2.0
        target_x = (W - 1) / 2.0
        dy = (target_y - com_y).view(B)
        dx = (target_x - com_x).view(B)

    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=K_grid.device, dtype=K_grid.dtype),
        torch.linspace(-1, 1, W, device=K_grid.device, dtype=K_grid.dtype),
        indexing='ij',
    )
    gy = gy.unsqueeze(0).expand(B, -1, -1).clone()
    gx = gx.unsqueeze(0).expand(B, -1, -1).clone()
    gy = gy - (dy * 2.0 / max(H - 1, 1)).view(B, 1, 1)
    gx = gx - (dx * 2.0 / max(W - 1, 1)).view(B, 1, 1)
    grid = torch.stack([gx, gy], dim=-1)
    # The same grid is applied to every channel of a batch item.
    K_centered = torch.nn.functional.grid_sample(
        K_grid, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
    )
    return K_centered.reshape(orig_shape)


def uncenter(img: FT, max_val: float = 1.0) -> FT:
    # [-max_val, max_val] -> [0, max_val]
    out = (img + max_val) * 0.5
    return out


def set_random_seed(*args):
    seed = hash(args) % (1 << 31)
    torch.manual_seed(seed)
    np.random.seed(seed)


def grad_norm(grads):
    return torch.sqrt(sum(
        (g**2).sum() for g in grads if g is not None
    ))


T = TypeVar("T", torch.Tensor, np.ndarray)

def equate_kernel_shapes(
    k1: T,
    k2: T
) -> Tuple[T, T]:
    if k1.ndim < 2:
        raise ValueError(f"Kernel k1 has {k1.ndim} dimensions, but expected 2. Shape: {k1.shape}")
    if k2.ndim < 2:
        raise ValueError(f"Kernel k2 has {k2.ndim} dimensions, but expected 2. Shape: {k2.shape}")
    if k1.ndim == 5 and k2.ndim == 4:
        # need to take center kernel for k1
        k1 = k1[:, k1.shape[1] // 2]
    elif k2.ndim == 5 and k1.ndim == 4:
        # need to take center kernel for k2
        k2 = k2[:, k2.shape[1] // 2]
    if k1.ndim != k2.ndim:
        raise ValueError(f"Kernels k1 and k2 must have the same number of dimensions. Found {k1.ndim} and {k2.ndim}")
    for dim in range(k1.ndim - 2):
        if k1.shape[dim] != k2.shape[dim]:
            raise ValueError(f"Kernels must have the same shape apart for the last two dimensions. "
                             f"Found shapes {k1.shape[dim]} and {k2.shape[dim]} at dimension {dim}")
    if k1.shape != k2.shape:
        # Expand the smaller kernel to the same size as the larger kernel with zeros
        k1_shp = k1.shape
        k2_shp = k2.shape
        if k1_shp[-1] < k2_shp[-1] and k1_shp[-2] > k2_shp[-2]:
            raise ValueError(
                f"Kernel k1 must be smaller or larger than k2"
                f" in both dimensions. Found k1 of shape {k1.shape}"
                f" and k2 of shape {k2.shape}"
            )
        if k2_shp[-1] < k1_shp[-1] and k2_shp[-2] > k1_shp[-2]:
            raise ValueError(
                f"Kernel k1 must be smaller or larger than k2"
                f" in both dimensions. Found k1 of shape {k1.shape}"
                f" and k2 of shape {k2.shape}"
            )
        revert = False
        if k2_shp[-1] < k1_shp[-1]:
            k1, k2 = k2, k1
            k1_shp, k2_shp = k2_shp, k1_shp
            revert = True
        # Now k1 < k2
        dim0_diff = k2_shp[-1] - k1_shp[-1]
        dim1_diff = k2_shp[-2] - k1_shp[-2]
        pad = [
            (dim0_diff // 2, dim0_diff // 2 + dim0_diff % 2),
            (dim1_diff // 2, dim1_diff // 2 + dim1_diff % 2),
        ]
        if isinstance(k1, torch.Tensor):
            # torch pad only pads the last dimensions so this should be good
            k1 = torch.nn.functional.pad(k1, [*pad[0], *pad[1]], value=0)
        else:
            np_pad = [(0, 0) for _ in range(k1.ndim - 2)] + pad
            k1 = np.pad(k1, np_pad, constant_values=0)
        if revert:
            k1, k2 = k2, k1
    return k1, k2




class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0, is_infinite: bool = True):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        warnings.filterwarnings('ignore', '`data_source` argument is not used and will be removed')
        super().__init__()
        self.dataset_size = len(dataset)
        self.start_idx = start_idx + rank
        self.stride = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.is_infinite = is_infinite

    def __iter__(self):
        idx = self.start_idx
        epoch = None
        while True:
            if epoch != idx // self.dataset_size:
                epoch = idx // self.dataset_size
                order = np.arange(self.dataset_size)
                if self.shuffle:
                    np.random.RandomState(hash((self.seed, epoch)) % (1 << 31)).shuffle(order)
            assert epoch is not None
            if epoch > 0 and not self.is_infinite:
                break
            yield int(order[idx % self.dataset_size])
            idx += self.stride

    def __len__(self) -> int:
        if self.is_infinite:
            return 1_000_000_000_000_000
        return self.dataset_size


def crop_valid(img: torch.Tensor, kernel_size: int | tuple[int, int], dims: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """Crop an image to the portion which is 'valid' when convolved with kernel of size `kernel_size`
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if kernel_size[0] == 0 and kernel_size[1] == 0:
        return img
    crop = (
        (math.ceil((kernel_size[0] + 1) / 2 - 1), img.shape[dims[0]] - (kernel_size[0] - 1)),
        (math.ceil((kernel_size[1] + 1) / 2 - 1), img.shape[dims[1]] - (kernel_size[1] - 1))
    )
    return img.narrow(dims[0], crop[0][0], crop[0][1]).narrow(dims[1], crop[1][0], crop[1][1])


def pad_valid(img: torch.Tensor, kernel_size: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    # padding is specified on width then height (different order)
    pad = (
        math.ceil((kernel_size[1] + 1) / 2 - 1), math.floor((kernel_size[1] + 1) / 2 - 1),
        math.ceil((kernel_size[0] + 1) / 2 - 1), math.floor((kernel_size[0] + 1) / 2 - 1),
    )
    return torch.nn.functional.pad(img, pad, mode="constant", value=0)


def pad_kernel(kernel: torch.Tensor, kernel_size: int | Tuple[int, int]) -> torch.Tensor:
    """
    Pad a kernel with zeros until it reaches the desired size while
    keeping the original kernel in the center.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if kernel.shape[-2] > kernel_size[0] or kernel.shape[-1] > kernel_size[1]:
        raise ValueError(
            f"Cannot pad kernel. Kernel larger than desired size. "
            f"Found kernel of shape {kernel.shape} and desired size {kernel_size}."
        )
    pad = (
        (kernel_size[0] - kernel.shape[-2]) // 2,
        (kernel_size[0] - kernel.shape[-2]) // 2 + (kernel_size[0] - kernel.shape[-2]) % 2,
        (kernel_size[1] - kernel.shape[-1]) // 2,
        (kernel_size[1] - kernel.shape[-1]) // 2 + (kernel_size[1] - kernel.shape[-1]) % 2,
    )
    return torch.nn.functional.pad(kernel, pad, value=0)


def get_center_kernel(kernels: torch.Tensor, img_h: int | None, img_w: int | None) -> torch.Tensor:
    if kernels.dim() == 5:
        img_hw = kernels.shape[1]
        if img_h is not None and img_w is not None:
            pass
        elif img_h is None and img_w is None:
            img_h = img_w = int(math.sqrt(img_hw))
        elif img_w is None:
            assert img_h is not None
            img_w = img_hw // img_h
        else:
            assert img_h is None and img_w is not None
            img_h = img_hw // img_w
        assert img_hw == img_h * img_w, f"{img_hw=} {img_h=} {img_w=}"

        kernels = kernels.view(-1, img_h, img_w, *kernels.shape[2:])
        kernels = kernels[:, img_h // 2, img_w // 2, ...]
    return kernels


# ----------------
# --- Image IO ---
# ----------------


def write_img_pt(img: torch.Tensor, path: str | Path | BytesIO, out_bits: int = 8) -> bool:
    """Write an image tensor (C,H,W, RGB, [0,1]) to a lossless PNG. out_bits=16
    keeps full precision (uint16, scaled by 65535); default 8 stays uint8."""
    img = img.clip(0, 1)
    out_dtype = torch.uint16 if int(out_bits) == 16 else torch.uint8
    img = TF.to_dtype(img, out_dtype, scale=True)
    # RGB->BGR channel swap. Use indexing, not .flip(): torch.flip has no CPU
    # kernel for uint16 (raises "flip_cpu not implemented for 'UInt16'").
    img = img[[2, 1, 0]]
    img = img.permute(1, 2, 0)
    img_np = img.numpy(force=True)
    # PNG is always lossless; set the zlib level explicitly (no quality loss).
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if isinstance(path, BytesIO):
        is_success, buffer = cv2.imencode(".png", img_np, png_params)
        path.write(buffer.tobytes())
        return is_success
    return cv2.imwrite(str(path), img_np, png_params)


# -------------------------------------
# --- Full images and image patches ---
# -------------------------------------

def pad_image_to_size(image, final_height, final_width, mode: str="replicate", value=None):
    image_height, image_width = image.shape[-2:]
    pad_params = [
        (final_width - image_width) // 2 if final_width > image_width else 0,
        (final_width - image_width + 1) // 2 if final_width > image_width else 0,
        (final_height - image_height) // 2 if final_height > image_height else 0,
        (final_height - image_height + 1) // 2 if final_height > image_height else 0,
    ]
    return torch.nn.functional.pad(image, pad_params, mode=mode, value=value)


def img2patches(img: torch.Tensor, patch_size: int, stride: int) -> list[torch.Tensor]:
    overlap = patch_size - stride
    img = pad_image_to_size(img, img.shape[-2] + overlap, img.shape[-1] + overlap)
    h, w = img.shape[-2:]
    w1 = list(np.arange(0, w-patch_size, stride, dtype=int))
    h1 = list(np.arange(0, h-patch_size, stride, dtype=int))
    w1.append(w-patch_size)
    h1.append(h-patch_size)
    patches = []
    for i in h1:
        for j in w1:
            patches.append(img[..., i:i+patch_size, j:j+patch_size].contiguous())
    return patches


def patches2img(
        patches: Sequence[torch.Tensor],
        stride: int, imgh: int, imgw: int, func: Callable[[torch.Tensor], torch.Tensor] | None) -> torch.Tensor:
    num_patches = len(patches)
    C, patch_size, _ = patches[0].shape
    dt, dev = patches[0].dtype, patches[0].device
    overlap = patch_size - stride
    w1 = list(np.arange(0, imgw-(patch_size - overlap), stride, dtype=int))
    h1 = list(np.arange(0, imgh-(patch_size - overlap), stride, dtype=int))
    w1.append(imgw-(patch_size - overlap))
    h1.append(imgh-(patch_size - overlap))
    assert num_patches == len(w1) * len(h1)
    img = torch.empty((C, imgh, imgw), dtype=dt, device=dev)
    pid = 0
    if func is None:
        func = lambda x: x
    for i in h1:
        for j in w1:
            cpatch = func(patches[pid])  # C, pS, pS
            img[..., i: i + (patch_size - overlap), j: j + (patch_size - overlap)].copy_(cpatch)
            pid += 1
    return img
