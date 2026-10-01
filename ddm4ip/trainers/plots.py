import math

import matplotlib
import matplotlib.axes
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms.v2.functional as TF
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from ddm4ip.data.base import Batch, DatasetType
from ddm4ip.data.utils import get_space_varying_patches
from ddm4ip.flowmatching_utils.generate import imggen
from ddm4ip.losses.base import AbstractLoss
from ddm4ip.losses.di_y import DiffInstructOnY
from ddm4ip.losses.di_x_y import DiffInstructOnXandY
from ddm4ip.networks.blur_nets import DegradationNetwork
from ddm4ip.utils import distributed
from ddm4ip.utils.metrics import calc_psnr
from ddm4ip.utils.torch_utils import equate_kernel_shapes, get_center_kernel


def imshow_with_cbar(fig, ax: matplotlib.axes.Axes, img, axis_off=True, cmap='viridis', vmin=None, vmax=None):
    img_shape = img.shape
    if len(img_shape) == 2:
        num_ch = 1
    else:
        num_ch = img_shape[-1]
    # if num_ch == 3
    plt_img = ax.imshow(img, aspect='equal', interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = num_ch == 1
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(plt_img, cax=cax, orientation='horizontal')
    if axis_off:
        ax.set_axis_off()


"""
Plot generated with diffusion
"""
@torch.no_grad()
def plot_imggen(config, flow_nn, batch: Batch, img_size: tuple[int, int, int], dset: DatasetType, device) -> torch.Tensor | None:
    if "generation" not in config:
        return None

    batch_size = config['training']['batch_size']
    is_space_varying = getattr(dset, "space_conditioning", False) or getattr(dset, "random_space_conditioning", False)
    if is_space_varying:
        all_patches = get_space_varying_patches(img_size=1024, patch_size=img_size[-1])
        n_h_patches, nrow = all_patches.shape[:2]
        all_patches = all_patches.view(-1, *all_patches.shape[2:])
        seeds = [42] * len(all_patches)
        conditioning = all_patches.to(device)
    else:
        num_img_for_sample = min(64, batch_size)
        seeds = range(42, 42 + num_img_for_sample)
        conditioning = batch.corrupt_conditioning[:num_img_for_sample].to(device) if batch.corrupt_conditioning is not None else None
        nrow = int(math.sqrt(num_img_for_sample))
        if conditioning is not None and (conditioning.shape[-2] != img_size[-2] or conditioning.shape[-1] != img_size[-1]):
            conditioning = TF.center_crop(conditioning, [img_size[-2], img_size[-1]])

    image_iter = imggen(
        net=flow_nn,
        img_size=img_size,
        label_dim=dset.label_dim,
        seeds=seeds,
        batch_size=batch_size,
        device=device,
        conditioning=conditioning,
        return_trajectory=False,
        **config['generation'],
    )
    # Gather within process
    gen_imgs = [img.solution for img in image_iter if img.solution is not None]
    gen_imgs = torch.cat(gen_imgs, dim=0) if len(gen_imgs) > 0 else None
    assert gen_imgs is not None
    # Gather across processes
    if torch.distributed.is_initialized():
        gen_imgs = torch.cat([img.cpu() for img in distributed._simple_gather_all_tensors(gen_imgs, None, distributed.get_world_size())], dim=0)
    image_grid = make_grid(
        gen_imgs.clamp_(0, 1), nrow=nrow, padding=0
    )
    return image_grid


def plot_imggen_iid(
    config,
    flow_nn,
    batch: Batch,
    img_size: tuple[int, int, int],
    dset: DatasetType,
    device,
    n_samples: int = 16,
) -> torch.Tensor | None:
    """Independent-sample diagnostic grid for space-conditioned flows.

    Companion to :func:`plot_imggen`. The default ``plot_imggen`` view for
    a space-conditioned dataset fixes the seed across all spatial
    positions to expose how the model's output varies with position.
    This complementary plot does the opposite: fixes the position
    (centre of a 1024x1024 face) and varies the seed across cells, so
    you can read off sample diversity at a single canonical location.

    Returns ``None`` (skips the plot) when the dataset is not
    space-conditioned, since ``plot_imggen`` already produces an iid grid
    in that case. Cost: ~``n_samples`` extra sampling chains per call
    (default 16). Plots run at ``plot_every_steps`` cadence, so this is
    negligible compared to training cost.
    """
    if "generation" not in config:
        return None
    is_space_varying = getattr(dset, "space_conditioning", False) or getattr(
        dset, "random_space_conditioning", False
    )
    if not is_space_varying:
        # plot_imggen already varies the seed in the non-space-varying case.
        return None

    # Build a fixed conditioning at the centre of the reference image.
    all_patches = get_space_varying_patches(img_size=1024, patch_size=img_size[-1])
    n_h, n_w = all_patches.shape[:2]
    centre = all_patches[n_h // 2, n_w // 2]  # (2, patch_size, patch_size)
    conditioning = (
        centre.unsqueeze(0)
        .repeat(n_samples, 1, 1, 1)
        .to(device)
    )

    seeds = list(range(42, 42 + n_samples))
    nrow = int(math.sqrt(n_samples))

    image_iter = imggen(
        net=flow_nn,
        img_size=img_size,
        label_dim=dset.label_dim,
        seeds=seeds,
        batch_size=config['training']['batch_size'],
        device=device,
        conditioning=conditioning,
        return_trajectory=False,
        **config['generation'],
    )
    gen_imgs = [img.solution for img in image_iter if img.solution is not None]
    gen_imgs = torch.cat(gen_imgs, dim=0) if len(gen_imgs) > 0 else None
    if gen_imgs is None:
        return None
    if torch.distributed.is_initialized():
        gen_imgs = torch.cat(
            [
                img.cpu()
                for img in distributed._simple_gather_all_tensors(
                    gen_imgs, None, distributed.get_world_size()
                )
            ],
            dim=0,
        )
    return make_grid(gen_imgs.clamp_(0, 1), nrow=nrow, padding=0)


"""
Plot explicit kernels
"""

def fix_kernel_dimensions(kernel, conditioning=None):
    """
    Fix kernel dimensions to be C, H, W. There are a variety of possible input dimensions.
    """
    if kernel.dim() == 2:
        kernel = kernel[None, :]  # kC, kH, kW
    elif kernel.dim() == 4:
        num_ch = kernel.shape[-3]
        if kernel.shape[0] > 1:
            kernel = make_grid(kernel, nrow=4, padding=0)
        else:
            kernel = kernel[0]
        if num_ch == 1:
            kernel = kernel[0:1]  # kC, H, W
    elif kernel.dim() == 5:
        # Multiple kernels for each image (B, H*W, kc, kh, kw). Visualize the middle one
        assert conditioning is not None
        num_ch = kernel.shape[-3]
        kernel = get_center_kernel(kernel, img_h=conditioning.shape[-2], img_w=conditioning.shape[-1])
        kernel = make_grid(kernel, nrow=4, padding=0)
        if num_ch == 1:
            kernel = kernel[0:1]  # kC, H, W
    return kernel


def kernel_diff(pred_kernel: torch.Tensor, true_kernel: torch.Tensor):
    # Now there are only two different possibilities based on number of kernel channels
    num_ch = pred_kernel.shape[0]
    if num_ch == 1:
        true_kernel_np = true_kernel[0].numpy(force=True)
        pred_kernel_np = pred_kernel[0].numpy(force=True)
        difference = pred_kernel_np - true_kernel_np
        mae = np.mean(np.abs(difference))
    elif num_ch == 3:
        true_kernel_np = true_kernel.permute(1, 2, 0).numpy(force=True)  # need channels-last for matplotlib
        pred_kernel_np = pred_kernel.permute(1, 2, 0).numpy(force=True)
        difference = np.mean(np.abs(pred_kernel_np - true_kernel_np), axis=-1)  # average of three channels
        mae = np.mean(difference)
    else:
        raise ValueError(f"Invalid number of channels: {num_ch}")
    return difference, mae


@torch.no_grad()
def plot_explicit_kernels(corruption, kernel_nn, batch: Batch | None, batch_size: int, device, center: bool = True):
    if not hasattr(kernel_nn.physics_net, "get_kernel"):
        return None, None
    true_ker_avail = corruption is not None and hasattr(corruption, "get_kernel")

    batch_dev = None
    if batch is not None:
        num_imgs = min(16, batch_size)
        batch_dev = batch[:num_imgs].to(device)
    conditioning = None
    if batch_dev is not None:
        conditioning = batch_dev.clean_conditioning

    # `center` only affects kernel nets that honour it (DirectKernel2 with
    # auto_center); others ignore it via **kwargs. center=False shows the
    # raw, pre-centering kernel for the drift diagnostic.
    pred_kernel = kernel_nn.physics_net.get_kernel(conditioning=conditioning, center=center)
    true_kernel = None   
    if true_ker_avail:
        true_kernel = corruption.get_kernel(conditioning=conditioning)
        pred_kernel, true_kernel = equate_kernel_shapes(pred_kernel, true_kernel)
        true_kernel = fix_kernel_dimensions(true_kernel, conditioning).cpu().detach()

    pred_kernel = fix_kernel_dimensions(pred_kernel, conditioning).cpu().detach()
    pred_filters_flat = pred_kernel.clone()

    # Display transform. Blur kernels sum to ~1 per channel, so their raw
    # values are tiny (peak ~0.05); shown as a float RGB image they render
    # near-black. For 3-channel (per-RGB / chromatic) kernels, normalise each
    # channel to [0,1] by its own max so the coloured blobs are visible. For
    # 1-channel kernels keep the previous channels-last array (viridis + cbar).
    def _disp(kernel_chw):
        a = kernel_chw.permute(1, 2, 0)
        if kernel_chw.shape[0] == 3:
            mx = a.reshape(-1, 3).amax(dim=0).clamp_min(1e-12)
            a = (a / mx).clamp(0, 1)
        return a
    rgb = pred_kernel.shape[0] == 3
    rgb_note = " (RGB, per-ch norm)" if rgb else ""

    if true_kernel is not None:
        difference, mae = kernel_diff(pred_kernel, true_kernel)
        fig, ax = plt.subplots(ncols=3, figsize=(9, 4))
        imshow_with_cbar(fig, ax[0], _disp(true_kernel))
        ax[0].set_title("True" + rgb_note)
        imshow_with_cbar(fig, ax[1], _disp(pred_kernel))
        ax[1].set_title("Pred" + rgb_note)
        imshow_with_cbar(fig, ax[2], difference)
        ax[2].set_title(f"Difference. MAE={mae:.3e}")
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        imshow_with_cbar(fig, ax, _disp(pred_kernel))
        ax.set_title("Predicted" + rgb_note)

    fig.tight_layout()
    return fig, pred_filters_flat


def make_kernels_grid(kernels: torch.Tensor, nrow: int) -> torch.Tensor | None:
    if kernels.dim() != 4:
        return None
    num_ch = kernels.shape[1]
    kernels = make_grid(kernels, nrow=nrow, padding=0)
    if num_ch == 1:
        kernels = kernels[0]
    else:
        kernels = kernels.permute(1, 2, 0)
    return kernels


@torch.no_grad()
def plot_spacevarying_kernels(
    corruption,
    patch_size: int,
    kernel_nn: torch.nn.Module,
    batch_size: int, device
):
    true_ker_avail = corruption is not None and hasattr(corruption, "get_kernel")

    # Choose the conditioning positions at which to query the kernel.
    # If the kernel net exposes a DISCRETE PSF grid (DirectCenterKernel: an
    # nH x nW grid of node kernels), sample AT the node positions so the montage
    # is exactly that grid (e.g. 8x8, the layout of the field). Each node is a
    # 1x1 conditioning map carrying its (y, x) position: get_kernel averages the
    # map to that position, and because pos coincides with the grid node the
    # interpolation returns the node kernel exactly. This is also memory-light
    # (one kernel per node) vs. sampling full patches. Otherwise (continuous /
    # per-pixel fields) fall back to a regular patch grid over a 1024px image.
    phys = getattr(kernel_nn, "physics_net", kernel_nn)
    psf_grid = getattr(phys, "psf_grid", None)
    if psf_grid is not None:
        n_nodes = psf_grid.shape[0]
        # Montage columns = number of distinct horizontal node positions
        # (num_psfs[1], = grid[:, 1]); rows = num_psfs[0]. The nodes are stored
        # row-major (k = i*num_psfs[1] + j), so make_grid(nrow=num_psfs[1])
        # lays node (i, j) at (row i, col j). Works for non-square grids
        # (e.g. 12x16); for a square grid this equals sqrt(n_nodes) as before.
        n_w_patches = int(torch.unique(psf_grid[:, 1]).numel())
        all_patches = psf_grid.detach().reshape(n_nodes, 2, 1, 1).float()
    else:
        # nH, nW, 2, patch_size, patch_size
        all_patches = get_space_varying_patches(img_size=1024, patch_size=patch_size)
        _, n_w_patches = all_patches.shape[:2]
        all_patches = all_patches.view(-1, *all_patches.shape[2:])

    pred_filters, true_filters = [], []
    for i in range(0, len(all_patches), batch_size):
        batch = all_patches[i: i + batch_size].to(device)
        try:
            assert isinstance(kernel_nn, DegradationNetwork)
            pf = kernel_nn.physics_net.get_kernel(conditioning=batch).cpu()
        except Exception:
            return None, None
        pf = get_center_kernel(pf, img_h=batch.shape[-2], img_w=batch.shape[-1])
        if pf.dim() != 4 or pf.shape[0] != batch.shape[0]:
            return None, None
        if true_ker_avail:
            tf = corruption.get_kernel(batch.cpu())
            tf = get_center_kernel(tf, img_h=batch.shape[-2], img_w=batch.shape[-1])
            if tf.dim() != 4 or tf.shape[0] != batch.shape[0]:
                return None, None
            tf, pf = equate_kernel_shapes(tf, pf)
            true_filters.append(tf)
        pred_filters.append(pf)
    pred_filters = torch.cat(pred_filters, 0)
    pred_filters_flat = pred_filters.clone()
    pred_filters = make_kernels_grid(pred_filters, nrow=n_w_patches)
    if true_ker_avail:
        true_filters = torch.cat(true_filters, 0)
        true_filters = make_kernels_grid(true_filters, nrow=n_w_patches)

    assert pred_filters is not None

    # Per-channel display normalization for RGB kernel grids. Raw kernel values
    # sum to ~1 over kh*kw, so their peak is tiny (~1e-2) and a float RGB image
    # renders near-black. Normalise by a SHARED per-channel max (over pred AND
    # true) so the coloured blobs are visible and the two panels stay comparable.
    def _disp(grid):
        return grid
    if pred_filters.ndim == 3 and pred_filters.shape[-1] == 3:
        mx = pred_filters.reshape(-1, 3).amax(0)
        if true_ker_avail and isinstance(true_filters, torch.Tensor):
            mx = torch.maximum(mx, true_filters.reshape(-1, 3).amax(0))
        mx = mx.clamp_min(1e-12)
        def _disp(grid):
            return (grid / mx).clamp(0, 1)

    if not true_ker_avail:
        fig, ax = plt.subplots(figsize=(5, 5))
        imshow_with_cbar(fig, ax, _disp(pred_filters))
        ax.set_title("Predicted")
    else:
        assert isinstance(true_filters, torch.Tensor)
        fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
        imshow_with_cbar(fig, ax[0], _disp(pred_filters))
        ax[0].set_title("Predicted")
        imshow_with_cbar(fig, ax[1], _disp(true_filters))
        ax[1].set_title("True")
        diff = (true_filters - pred_filters).abs()
        # MAE on raw values; the diff image averages channels for RGB.
        diff_disp = diff.mean(-1) if (diff.ndim == 3 and diff.shape[-1] == 3) else diff
        imshow_with_cbar(fig, ax[2], diff_disp)
        ax[2].set_title(f"Difference. MAE={diff.mean():.2e}")

    fig.tight_layout()
    return fig, pred_filters_flat


"""
Plot implicit kernels
"""

@torch.no_grad()
def plot_implicit_kernels(batch: Batch, kernel_nn, loss: AbstractLoss, num_imgs: int = 16):
    batch_dev = batch[:num_imgs].cuda()

    if batch_dev.clean is None or batch_dev.corrupt is None:
        return None  # dataset is not paired, can't do this plot...
    noise_params = batch_dev.noise_params
    if isinstance(loss, (DiffInstructOnY)):
        clean, clean_lbl = batch_dev.clean, r"$x$"
        corrupt_true, corrupt_true_lbl = batch_dev.corrupt, r"$y$"
        cond = batch_dev.clean_conditioning if batch_dev.clean_conditioning is not None else None
        corrupt_pred = loss.get_images(clean, conditioning=cond, return_physics=False, **noise_params)
        corrupt_pred_lbl = r"$A_{\hat{\omega}} x + \epsilon$"
        title = r"$y - A_{\hat{\omega}} x$"
    elif isinstance(loss, DiffInstructOnXandY):
        # Input PSNR(y, x) only makes sense when y and x are the same size
        # (deblurring). For super-resolution / downsampling y is smaller than x,
        # so skip it rather than crash on the size mismatch.
        if batch_dev.corrupt.shape == batch_dev.clean.shape:
            psnr_y = calc_psnr(batch_dev.corrupt, batch_dev.clean)
            clean, clean_lbl = batch_dev.corrupt, rf"$y$ PSNR: {psnr_y:.2f}"
        else:
            clean, clean_lbl = batch_dev.corrupt, r"$y$"
        corrupt_true, corrupt_true_lbl = batch_dev.clean, r"$x$"
        cond = batch_dev.corrupt_conditioning if batch_dev.corrupt_conditioning is not None else None
        corrupt_pred_lbl = r"$h_{\hat{\omega}}(A^*x + \epsilon)$"
        corrupt_pred = loss.get_images(kernel_nn, clean, conditioning=cond, return_physics=False, **noise_params)
        assert isinstance(corrupt_pred, torch.Tensor)
        psnr = calc_psnr(corrupt_true, corrupt_pred)
        title = r"$x - h_{\hat{\omega}}(y)$ " + f"PSNR: {psnr:.2f}"
    else:
        return None

    batch_size = clean.shape[0]
    n_img_row = int(math.sqrt(batch_size))

    clean = make_grid(
        clean.clamp_(0, 1), nrow=n_img_row, padding=0,
    ).permute(1, 2, 0).numpy(force=True)
    corrupt_true = make_grid(
        corrupt_true.clamp_(0, 1), nrow=n_img_row, padding=0,
    ).permute(1, 2, 0).numpy(force=True)
    corrupt_pred = make_grid(
        corrupt_pred.clamp_(0, 1), nrow=n_img_row, padding=0,
    ).permute(1, 2, 0).numpy(force=True)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 16))

    imshow_with_cbar(fig, ax[0, 0], clean)
    ax[0, 0].set_title(clean_lbl)

    imshow_with_cbar(fig, ax[0, 1], corrupt_true)
    ax[0, 1].set_title(corrupt_true_lbl)

    imshow_with_cbar(fig, ax[1, 0], corrupt_pred)
    ax[1, 0].set_title(corrupt_pred_lbl)

    if corrupt_true.shape == corrupt_pred.shape:
        diff = (corrupt_true - corrupt_pred).mean(2)  # average over channels
        if np.abs(diff).max() < 0.1:
            diff *= 10
            title = rf"{title} (x10)"
        ax[1, 1].imshow(diff, aspect='equal', interpolation='none', cmap='RdBu', vmin=-1.0, vmax=1.0)
        ax[1, 1].set_title(title)
    ax[1, 1].set_axis_off()

    fig.tight_layout()
    return fig


