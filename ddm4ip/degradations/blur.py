import deepinv
import numpy
import torch

from .degradation import fix_deepinv_state




def get_motion_blur_kernel(size: int, intensity: float, rnd_seed: int):
    from .motion_blur import Kernel
    numpy.random.seed(rnd_seed)
    mb_ker_obj = Kernel(size=(size, size), intensity=intensity)
    mb_ker = torch.from_numpy(mb_ker_obj.kernelMatrix).to(dtype=torch.float32)
    return mb_ker.view(1, 1, size, size)


def get_disk_blur_kernel(size: int, radius: float, edge: float = 0.7):
    """Disk / defocus (out-of-focus) PSF: a flat disk of given ``radius``.

    Models the geometric out-of-focus blur of a circular aperture —
    ``k(u) ∝ 1[‖u‖ ≤ radius]`` — normalised to sum to one. The hard
    circular edge is anti-aliased with a smooth-step over a band of width
    ``edge`` pixels (a soft coverage fraction per pixel), so the kernel
    isn't a jagged binary mask and remains well-defined for fractional
    radii. ``edge → 0`` recovers the sharp binary disk.

    Returns a ``(1, 1, size, size)`` float32 tensor.
    """
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    if edge <= 0:
        disk = (r <= radius).to(torch.float32)
    else:
        # Linear ramp from 1 (inside) to 0 (outside) across the edge band,
        # clamped to [0, 1]: smooth, monotone coverage fraction.
        disk = torch.clamp((radius - r) / edge + 0.5, 0.0, 1.0)
    disk = disk / disk.sum().clamp_min(1e-12)
    return disk.view(1, 1, size, size)


def get_aniso_gaussian_kernel(
    size: int,
    sigma_x: float,
    sigma_y: float,
    angle: float = 0.0,
    shift: tuple[float, float] = (0.0, 0.0),
):
    """Single-channel anisotropic (elliptical) Gaussian PSF.

    ``sigma_x``/``sigma_y`` are the std-devs along the ellipse's two principal
    axes (in pixels), ``angle`` rotates the ellipse counter-clockwise (degrees).
    ``shift = (shift_x, shift_y)`` offsets the PSF peak from the kernel centre by
    that many pixels; since ``(f * k)(p) = f(p - q0)`` for a kernel centred at
    ``q0``, this *translates* the convolved channel by ``(shift_x, shift_y)`` —
    used to simulate per-channel displacement (chromatic aberration).

    Returns a ``(1, 1, size, size)`` float32 tensor summing to one.
    """
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xx = xx - float(shift[0])
    yy = yy - float(shift[1])
    theta = torch.deg2rad(torch.tensor(angle, dtype=torch.float32))
    cos, sin = torch.cos(theta), torch.sin(theta)
    # rotate the (shifted) coordinate frame into the ellipse axes
    xr = cos * xx + sin * yy
    yr = -sin * xx + cos * yy
    g = torch.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    g = g / g.sum().clamp_min(1e-12)
    return g.view(1, 1, size, size)


def get_rgb_aniso_gaussian_kernel(
    size: int,
    sigma_x,
    sigma_y,
    angle=0.0,
    shift=None,
):
    """Per-RGB-channel anisotropic Gaussian PSF -> ``(1, 3, size, size)``.

    Each of ``sigma_x``, ``sigma_y``, ``angle`` may be a scalar (shared across
    channels) or a length-3 sequence (one value per R, G, B channel). ``shift``
    is either ``None`` (no displacement) or a length-3 sequence of
    ``(shift_x, shift_y)`` per-channel offsets — non-zero shifts simulate
    lateral chromatic aberration. deepinv's ``conv2d`` convolves channel ``c``
    of the image with channel ``c`` of this filter (per-channel blur).
    """
    def _triplet(v):
        if isinstance(v, (int, float)):
            return [float(v)] * 3
        v = list(v)
        assert len(v) == 3, f"expected scalar or length-3, got {v}"
        return [float(x) for x in v]

    sx, sy, ang = _triplet(sigma_x), _triplet(sigma_y), _triplet(angle)
    if shift is None:
        shifts = [(0.0, 0.0)] * 3
    else:
        shifts = [tuple(float(x) for x in s) for s in shift]
        assert len(shifts) == 3, f"shift must have 3 entries, got {shift}"
    chans = [
        get_aniso_gaussian_kernel(size, sx[c], sy[c], ang[c], shifts[c])
        for c in range(3)
    ]
    return torch.cat(chans, dim=1)  # (1, 3, size, size)


class Blur(deepinv.physics.Blur):
    """Wraps deepinv `Blur`"""
    def get_kernel(self, img=None, conditioning=None):
        return self.filter

    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d

class Downsampling(deepinv.physics.Downsampling):
    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d

    def get_kernel(self, img=None, conditioning=None):
        return self.filter


class Identity(deepinv.physics.LinearPhysics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def A(self, x, **kwargs):
        return x

    def A_adjoint(self, y, **kwargs):
        return y

    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d
