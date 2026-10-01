import abc
import warnings
from enum import Enum
from typing import Literal
import deepinv
import torch
import torch.nn as nn

from ddm4ip.degradations.blur import Blur
from ddm4ip.psf.psf import (
    PSF,
    apply_kernel_norm,
    get_psf_at_pos,
    norm_smooth,
    norm_sum_to_one,
)
from ddm4ip.utils import distributed, persistence
from ddm4ip.utils.torch_utils import center_kernel_via_shift, windowed_kernel_com


def _apply_norm(x: torch.Tensor, kernel_norm: str | None) -> torch.Tensor:
    """``kernel_norm`` by name (see ``apply_kernel_norm``); ``norm_smooth`` if unset."""
    if kernel_norm is not None:
        return apply_kernel_norm(x, kernel_norm)
    return norm_smooth(x)


class BlurDegradationType(Enum):
    BLUR = 1
    DOWNSAMPLING = 3


class NormalizedDownsampling(deepinv.physics.Downsampling):
    r"""Zero-padded downsampling, renormalised by the kernel mass that actually
    landed on real pixels (normalized convolution):

    .. math::
        A(x) = D\left[ \frac{(x \odot m) * k}{m * k} \right], \qquad m = \mathbb{1}

    Interior outputs are unchanged (a sum-to-one kernel gives ``m * k = 1``
    there); border outputs are divided by the fraction of kernel mass that
    overlapped the image, so no boundary content is invented and the output size
    (hence the decimation phase) is the same as with circular padding. The
    operator stays linear in ``x`` (the denominator depends only on the mask and
    the filter), and the normalisation stays in the autograd graph.
    """

    def __init__(self, *args, eps: float = 1e-6, **kwargs):
        kwargs["padding"] = "constant"          # true zero padding
        super().__init__(*args, **kwargs)
        self.eps = eps

    def _mask_response(self, x: torch.Tensor) -> torch.Tensor:
        # m * k through the same code path as the signal.
        return super().A(torch.ones_like(x))

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y = super().A(x, **kwargs)              # updates the filter from kwargs
        return y / (self._mask_response(x) + self.eps)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # (diag(1/n) . B)^T = B^T . diag(1/n); n has the measurement's shape.
        n = super().A(torch.ones(
            y.shape[0], y.shape[1],
            y.shape[-2] * self.factor, y.shape[-1] * self.factor,
            device=y.device, dtype=y.dtype,
        ))
        return super().A_adjoint(y / (n + self.eps), **kwargs)


class AbstractBlurNet(nn.Module, abc.ABC):
    def __init__(
        self,
        blur_type: BlurDegradationType,
        padding: str,
        **blur_kwargs,
    ):
        super().__init__()

        self.degradation_type = blur_type
        self.padding = padding
        self.blur_kwargs = blur_kwargs

    def get_physics(
        self,
        crop_filters: int = 0,
        **kwargs
    ) -> deepinv.physics.Physics:
        """
        Args:
            crop_filters (int, optional): crop this many pixels from the border of
                the kernels (and renormalise). Used at evaluation time. Defaults to 0.
        """
        degradation: deepinv.physics.Blur | deepinv.physics.Downsampling
        if self.degradation_type == BlurDegradationType.BLUR:
            degradation = Blur(padding=self.padding, device="cuda", **self.blur_kwargs)
        elif self.degradation_type == BlurDegradationType.DOWNSAMPLING:
            # "zeros_normalized" is not a deepinv padding mode.
            if self.padding == "zeros_normalized":
                degradation = NormalizedDownsampling(device="cuda", **self.blur_kwargs)
            else:
                degradation = deepinv.physics.Downsampling(padding=self.padding, device="cuda", **self.blur_kwargs)
        else:
            raise NotImplementedError(self.degradation_type)

        kernels = self.get_kernel(**kwargs)

        if crop_filters > 0:
            kernels = kernels[..., crop_filters:-crop_filters, crop_filters:-crop_filters]
            kernels = norm_sum_to_one(kernels)
        # The filter is set after construction because `Blur` would reset
        # `requires_grad` to False.
        degradation.filter = kernels
        return degradation

    @abc.abstractmethod
    def get_kernel(self, **kwargs) -> torch.Tensor:
        pass


def _center(K: torch.Tensor, window: float, anchor_channel: int | None) -> torch.Tensor:
    """Sub-pixel, differentiable recentering of the windowed centre of mass."""
    return center_kernel_via_shift(K, bg_quantile=0.5, window=window, anchor_channel=anchor_channel)


def _com_shift(K: torch.Tensor, window: float, anchor_channel: int):
    """Integer shift that brings the anchor channel's windowed centre of mass
    to the geometric centre, one (dy, dx) per leading kernel."""
    H, W = K.shape[-2], K.shape[-1]
    cy, cx = windowed_kernel_com(K, window=window, bg_quantile=0.5)
    cy = cy[:, anchor_channel].reshape(-1)
    cx = cx[:, anchor_channel].reshape(-1)
    return [(int(round((H - 1) / 2.0 - float(y))), int(round((W - 1) / 2.0 - float(x))))
            for y, x in zip(cy, cx)]


def _roll_param_(param: torch.Tensor, shifts, optimizer: torch.optim.Optimizer | None):
    """Roll each leading entry of `param` (and its Adam moments) in place."""
    st = optimizer.state.get(param) if optimizer is not None else None
    for n, (dy, dx) in enumerate(shifts):
        if dy == 0 and dx == 0:
            continue
        param[n].copy_(torch.roll(param[n], shifts=(dy, dx), dims=(-2, -1)))
        if st:
            for buf_key in ("exp_avg", "exp_avg_sq"):
                if buf_key in st and torch.is_tensor(st[buf_key]):
                    st[buf_key][n].copy_(torch.roll(st[buf_key][n], shifts=(dy, dx), dims=(-2, -1)))


@persistence.persistent_class
class DirectKernel2(AbstractBlurNet):
    """A single learnable kernel (per channel when ``kernel_ch > 1``), followed by
    decimation when ``factor > 1``.

    With ``auto_center``, the raw parameter is recentred in place by an integer
    roll after every operator update (``recenter_kernel_inplace_``), and the
    residual sub-pixel offset is removed at the output of ``get_kernel``. For a
    multi-channel kernel, one shift computed on ``auto_center_anchor_channel`` is
    applied to all channels, which preserves inter-channel offsets.
    """

    def __init__(
        self,
        kernel_size: int,
        padding: str,
        factor: int = 1,
        kernel_norm: str | None = None,
        kernel_ch: int = 1,
        auto_center: bool = False,
        auto_center_window: float = 8.0,
        auto_center_anchor_channel: int = 0,
        initialization: str = "random",
    ):
        blur_type = BlurDegradationType.BLUR if factor == 1 else BlurDegradationType.DOWNSAMPLING
        # `factor` is ignored by deepinv when not downsampling.
        super().__init__(blur_type=blur_type, padding=padding, factor=factor)
        self.kernel_norm = kernel_norm
        self.auto_center = bool(auto_center)
        self.auto_center_window = float(auto_center_window)
        self.kernel_ch = int(kernel_ch)
        self.auto_center_anchor_channel = max(
            0, min(int(auto_center_anchor_channel), self.kernel_ch - 1)
        )
        if initialization != "random":
            raise NotImplementedError(f"DirectKernel2: initialization '{initialization}' not implemented.")
        # A softplus parameter takes both signs; otherwise the parameter
        # is positive and normalised with norm_smooth.
        smooth_norm = kernel_norm == "softplus_sum_to_one"
        kernel = torch.randn(1, self.kernel_ch, kernel_size, kernel_size)
        if not smooth_norm:
            kernel = norm_smooth(kernel.abs_())
        self.kernel = torch.nn.Parameter(kernel, requires_grad=True)

    def get_kernel(self, center: bool = True, **kwargs):
        """The forward kernel; ``center=False`` skips the sub-pixel recentering."""
        do_center = center and self.auto_center
        if do_center and self.kernel_norm == "softplus_sum_to_one":
            # softplus, then recenter, then normalise: mass shifted off-frame by
            # the recentering must not break the sum-to-one constraint.
            K = torch.nn.functional.softplus(self.kernel)
            K = _center(K, self.auto_center_window, self._center_anchor())
            return norm_sum_to_one(K)
        K = _apply_norm(self.kernel, self.kernel_norm)
        if do_center:
            K = _center(K, self.auto_center_window, self._center_anchor())
        return K

    def _center_anchor(self):
        return self.auto_center_anchor_channel if self.kernel_ch > 1 else None

    @torch.no_grad()
    def recenter_kernel_inplace_(self, optimizer: torch.optim.Optimizer | None = None):
        """Integer-roll the raw parameter so that its windowed centre of mass (on
        softplus of the parameter) sits at the geometric centre. Called once per
        operator update, after the optimizer step; the Adam moments are rolled
        by the same shift."""
        if not self.auto_center:
            return
        K = torch.nn.functional.softplus(self.kernel)  # (1, C, H, W)
        shifts = _com_shift(K, self.auto_center_window, self.auto_center_anchor_channel)
        _roll_param_(self.kernel, shifts, optimizer)

    # The normalisation and the centering flag are stored with the parameter, so
    # that a loaded checkpoint is evaluated with the forward it was trained with.
    def get_extra_state(self):
        return {
            "kernel_norm": self.kernel_norm,
            "auto_center": self.auto_center,
        }

    def set_extra_state(self, state):
        if not isinstance(state, dict) or not state:
            return
        if "kernel_norm" in state and state["kernel_norm"] != self.kernel_norm:
            warnings.warn(
                f"DirectKernel2: the checkpoint was trained with kernel_norm="
                f"{state['kernel_norm']!r}, the config has {self.kernel_norm!r}; "
                f"using the checkpoint's.",
                stacklevel=2,
            )
            self.kernel_norm = state["kernel_norm"]
        if "auto_center" in state and state["auto_center"] != self.auto_center:
            warnings.warn(
                f"DirectKernel2: the checkpoint was trained with auto_center="
                f"{state['auto_center']}, the config has {self.auto_center}; "
                f"using the checkpoint's.",
                stacklevel=2,
            )
            self.auto_center = bool(state["auto_center"])


@persistence.persistent_class
class DirectCenterKernel2(AbstractBlurNet):
    """Space-varying operator: a grid of kernels, interpolated at the centre of
    each patch (read from the location conditioning). With
    ``initialization="from_psf"`` the grid is the ground-truth PSF grid read from
    ``psf_path`` (oracle). Centering is as in ``DirectKernel2``, per grid entry.
    """

    def __init__(
        self,
        psf_path: str | None,
        kernel_size: int,
        padding: str,
        num_psfs: tuple[int, int] | None = None,
        kernel_ch: int | None = None,
        initialization: Literal["random", "from_psf"] = "random",
        kernel_norm: str | None = None,
        auto_center: bool = False,
        auto_center_window: float = 8.0,
        auto_center_anchor_channel: int = 0,
    ):
        super().__init__(BlurDegradationType.BLUR, padding=padding)
        self.kernel_norm = kernel_norm
        self.auto_center = bool(auto_center)
        self.auto_center_window = float(auto_center_window)
        loaded_psfs = None
        if psf_path is None:
            assert num_psfs is not None, "psf_path is None so num_psfs must be specified"
            assert kernel_ch is not None, "psf_path is None so kernel_ch must be specified"
            psf_grid = get_default_psf_grid(num_psfs)
            psf_shape = (psf_grid.shape[0], kernel_ch, kernel_size, kernel_size)
        else:
            psf = PSF.from_path(
                psf_path=psf_path, filter_size=kernel_size
            )
            psf_grid = psf.loc
            psf_shape = psf.psfs.shape
            loaded_psfs = psf.psfs
        self.psf_grid = torch.nn.Buffer(psf_grid)
        self.kernel_ch = int(psf_shape[1])
        self.auto_center_anchor_channel = max(
            0, min(int(auto_center_anchor_channel), self.kernel_ch - 1)
        )

        if initialization == "random":
            empty_kernels = torch.randn(psf_shape).abs_()
        elif initialization == "from_psf":
            if loaded_psfs is None:
                raise ValueError(
                    "initialization='from_psf' requires psf_path (the ground-truth PSF grid)."
                )
            empty_kernels = loaded_psfs.clone().float()
        else:
            raise NotImplementedError(
                f"initialization {initialization} not implemented for DirectCenterKernel network."
            )
        empty_kernels = norm_smooth(empty_kernels, eps=1e-5)
        self.kernels = torch.nn.Parameter(empty_kernels, requires_grad=True)

        distributed.print0("Initialized DirectCenterKernel model:")
        distributed.print0(f"Direct spatially-varying PSF shape:    {self.kernels.shape}")
        distributed.print0(f"Padding:                               {self.padding}")
        distributed.print0(f"Initialization:                        {initialization}")
        distributed.print0(f"Auto-center:                           {self.auto_center} "
                           f"(anchor_ch={self.auto_center_anchor_channel})")
        distributed.print0()

    def _center_anchor(self):
        return self.auto_center_anchor_channel if self.kernel_ch > 1 else None

    def get_kernel(self, conditioning: torch.Tensor | None = None,
                   center: bool = True, **kwargs):
        # conditioning: B, 2(xy), H, W
        if conditioning is None:
            raise ValueError("conditioning cannot be None for DirectCenterKernel")
        if conditioning.shape[1] != 2:
            raise ValueError(f"DirectCenterKernel expects conditioning with 2 channels but found {conditioning.shape[1]}")

        # Kernel at the centre of each patch.
        centers = conditioning.mean(dim=(-1, -2))  # B, 2
        interp_kernels = get_psf_at_pos(
            psfs=self.kernels,
            grid=self.psf_grid,
            pos=centers,
            norm_output=False,
        )
        do_center = center and self.auto_center
        if do_center and self.kernel_norm == "softplus_sum_to_one":
            K = torch.nn.functional.softplus(interp_kernels)
            K = _center(K, self.auto_center_window, self._center_anchor())
            return norm_sum_to_one(K)
        interp_kernels = _apply_norm(interp_kernels, self.kernel_norm)
        if do_center:
            interp_kernels = _center(interp_kernels, self.auto_center_window, self._center_anchor())
        return interp_kernels  # shape  B, kc, kh, kw

    @torch.no_grad()
    def recenter_kernel_inplace_(self, optimizer: torch.optim.Optimizer | None = None):
        """In-place integer recentering of each grid entry (see DirectKernel2)."""
        if not self.auto_center:
            return
        K = torch.nn.functional.softplus(self.kernels)  # (N, C, H, W)
        shifts = _com_shift(K, self.auto_center_window, self.auto_center_anchor_channel)
        _roll_param_(self.kernels, shifts, optimizer)


def get_default_psf_grid(num_psfs: tuple[int, int]):
    return torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, num_psfs[0]),  # x
            torch.linspace(0, 1, num_psfs[1]),  # y
            indexing='ij'
        ),
        dim=-1
    ).view(-1, 2)


@persistence.persistent_class
class DegradationNetwork(nn.Module):
    """Learnable operator followed by the measurement noise model."""

    def __init__(
        self,
        physics_network: AbstractBlurNet,
        noise_network: nn.Module,
    ):
        super().__init__()
        self.physics_net = physics_network
        self.noise_net = noise_network

    def forward(self, clean_img: torch.Tensor, return_physics: bool = False, **kwargs) -> torch.Tensor | tuple[torch.Tensor, deepinv.physics.Physics]:
        physics = self.get_physics(**kwargs)
        out = physics(clean_img)
        if return_physics:
            return out, physics
        return out

    def get_physics(self, **kwargs) -> deepinv.physics.Physics:
        physics = self.physics_net.get_physics(**kwargs)
        noise_model = self.noise_net.get_noisemodel(**kwargs)
        physics.noise_model = noise_model
        return physics

    def recenter_kernel_inplace_(self, optimizer=None):
        fn = getattr(self.physics_net, "recenter_kernel_inplace_", None)
        if fn is not None:
            fn(optimizer)
