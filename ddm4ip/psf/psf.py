import torch
import torchvision.transforms.v2.functional as TF


def norm_sum_to_one(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    return x / (x.sum(dim=(-1, -2), keepdim=True) + eps)


def norm_smooth(x, eps=1e-6):
    x = torch.abs(x) / (torch.sqrt(1 + x**2))  # bounds in [-1, 1]
    return x / (x.sum(dim=(-1, -2), keepdim=True) + eps)


def norm_softplus_sum_to_one(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Softplus, then division by the per-kernel sum (last two dimensions)."""
    x = torch.nn.functional.softplus(x)
    return x / (x.sum(dim=(-1, -2), keepdim=True) + eps)


def norm_identity(x: torch.Tensor) -> torch.Tensor:
    return x


_KERNEL_NORM_REGISTRY = {
    "softplus_sum_to_one": norm_softplus_sum_to_one,
    "none": norm_identity,
}


def apply_kernel_norm(x: torch.Tensor, name: str) -> torch.Tensor:
    """Dispatch a normalization by name. Raises if name is unknown."""
    try:
        return _KERNEL_NORM_REGISTRY[name](x)
    except KeyError as e:
        valid = ", ".join(sorted(_KERNEL_NORM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown kernel_norm '{name}'. Valid options: {valid}"
        ) from e


def get_psf_at_pos(
    psfs: torch.Tensor,
    grid: torch.Tensor,
    pos: torch.Tensor,
    norm_output: bool = True,
) -> torch.Tensor:
    """Bilinear interpolation of a regular grid of PSFs.

    psfs: H*W, kc, kh, kw
    grid: H*W, 2
    pos: B, 2
    output: B, kc, kh, kw
    """
    x = torch.unique(grid[:, 0])
    y = torch.unique(grid[:, 1])

    psf_shape = psfs.shape[1:]
    psfs = psfs.view(len(y), len(x), -1).permute(2, 0, 1)[None, :]

    num_queries = pos.shape[0]
    pos = pos.view(1, num_queries, 1, 2).to(torch.float32) * 2 - 1
    with torch.backends.cudnn.flags(enabled=False):
        interp = torch.nn.functional.grid_sample(psfs.contiguous(), pos.to(psfs.device).contiguous(), align_corners=True)
    interp = interp.view(-1, num_queries).T.reshape(num_queries, *psf_shape)
    if norm_output:
        # Without gradient (dataset degradation, evaluation), normalise in place
        # to avoid a second full-size allocation when one PSF is queried per pixel.
        if torch.is_grad_enabled():
            return norm_sum_to_one(interp, eps=1e-5)
        interp /= interp.sum(dim=(-1, -2), keepdim=True) + 1e-5
        return interp
    return interp


class PSF(torch.nn.Module):
    """A set of PSFs (normalised to sum to one) at positions in [0, 1]^2."""

    def __init__(
        self,
        positions: torch.Tensor,
        kernels: torch.Tensor,
    ):
        super().__init__()
        if kernels.dim() != 4:
            raise ValueError(f"Kernels must have 4 dimensions: B, C, H, W but found shape {kernels.shape}.")
        if positions.shape[0] != kernels.shape[0]:
            raise ValueError(f"Positions and kernels must have the same batch dimension but found {positions.shape[0]} and {kernels.shape[0]}.")
        if positions.dim() != 2 or positions.shape[1] != 2:
            raise ValueError(f"Positions must have shape B, 2 for x, y positions but found shape {positions.shape}.")
        if positions.amin() < 0 or positions.amax() > 1:
            raise ValueError(f"Positions should be within the [0, 1] range but found {positions.amin():.4f} and {positions.amax():.4f}")
        self.filter_size = (kernels.shape[-2], kernels.shape[-1])
        self.N = positions.shape[0]

        self.loc = torch.nn.Buffer(positions)
        self.psfs = torch.nn.Buffer(norm_sum_to_one(kernels))

    def __len__(self):
        return self.N

    @staticmethod
    def from_path(psf_path, filter_size=None):
        psf_data = torch.load(psf_path, weights_only=True)

        psfs = psf_data["psf"]
        if filter_size is not None:
            psfs = TF.center_crop(psfs, filter_size)

        return PSF(
            torch.stack([psf_data["x"], psf_data["y"]], 1),
            psfs,
        )
