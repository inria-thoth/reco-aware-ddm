from pathlib import Path
import deepinv
import scipy
import scipy.io
import torch

from ddm4ip.utils.torch_utils import pad_kernel

class GaussianNoise(deepinv.physics.GaussianNoise):
    """Wraps deepinv `GaussianNoise`"""
    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d


class ZeroNoise(deepinv.physics.ZeroNoise):
    """Wraps deepinv `ZeroNoise`"""
    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d


def init_noise(noise_cfg):
    noise_kind = noise_cfg.get("kind", "none")
    if noise_kind == "gaussian":
        noise_model = GaussianNoise(sigma=noise_cfg["std"])
    elif noise_kind == "none":
        noise_model = ZeroNoise()
    else:
        raise ValueError(f"noise '{noise_kind}' not recognized")
    return noise_model


def load_filter_from_file(path, kernel_size: int | None):
    path = Path(path)
    if path.suffix == ".mat":
        kernel = scipy.io.loadmat(path)["Kernel"]
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel[None, None, ...]
    else:
        raise NotImplementedError("Only mat files are supported")
    if kernel_size is not None:
        kernel = pad_kernel(kernel, kernel_size)
    return kernel


def instantiate_single_kernel(pert_cfg, noise_model, device):
    pert_kind = pert_cfg.get("kind", "none")
    if pert_kind == "gaussian_blur":
        from ddm4ip.degradations.blur import Blur
        filter = deepinv.physics.blur.gaussian_blur(
            sigma=pert_cfg['kernel_std'],
            angle=pert_cfg.get('kernel_angle', 0),
        )
        return Blur(
            filter=filter,
            padding=pert_cfg["padding"],
            noise_model=noise_model,
            device = device
        )
    elif pert_kind == "aniso_gaussian_blur":
        # Anisotropic (elliptical) Gaussian: distinct std along the two axes,
        # rotated by `kernel_angle` degrees. The standard SR / deblurring
        # kernel family (e.g. DIV2KRK). deepinv's gaussian_blur takes a
        # (sigma_x, sigma_y) tuple + angle.
        from ddm4ip.degradations.blur import Blur
        filter = deepinv.physics.blur.gaussian_blur(
            sigma=(pert_cfg['kernel_std_x'], pert_cfg['kernel_std_y']),
            angle=pert_cfg.get('kernel_angle', 0),
        )
        return Blur(
            filter=filter,
            padding=pert_cfg["padding"],
            noise_model=noise_model,
            device = device
        )
    elif pert_kind == "file_downsampling":
        from ddm4ip.degradations.blur import Downsampling
        filter = load_filter_from_file(pert_cfg["kernel_path"], kernel_size=pert_cfg.get("kernel_size", None))
        return Downsampling(
            img_size=None,
            filter=filter, # pyright: ignore[reportArgumentType]
            factor=pert_cfg["factor"],
            padding=pert_cfg["padding"],
            noise_model=noise_model,
            device = device
        )
    elif pert_kind == "rgb_aniso_gaussian_blur":
        # Per-RGB-channel anisotropic Gaussian. Each channel gets its own
        # (sigma_x, sigma_y, angle); with non-zero per-channel `shift` it also
        # models lateral chromatic aberration (channels displaced relative to
        # each other). Filter shape (1, 3, k, k) -> deepinv convolves channel c
        # of the image with channel c of the filter.
        from ddm4ip.degradations.blur import Blur, get_rgb_aniso_gaussian_kernel
        filter = get_rgb_aniso_gaussian_kernel(
            size=pert_cfg["kernel_size"],
            sigma_x=pert_cfg["kernel_std_x"],
            sigma_y=pert_cfg["kernel_std_y"],
            angle=pert_cfg.get("kernel_angle", 0.0),
            shift=pert_cfg.get("kernel_shift", None),
        )
        return Blur(
            filter=filter,
            padding=pert_cfg["padding"],
            noise_model=noise_model,
            device=device,
        )
    elif pert_kind == "disk_blur":
        from ddm4ip.degradations.blur import Blur, get_disk_blur_kernel
        filter = get_disk_blur_kernel(
            size=pert_cfg["kernel_size"],
            radius=pert_cfg["radius"],
            edge=pert_cfg.get("edge", 0.7),
        )
        return Blur(
            filter=filter,
            padding=pert_cfg["padding"],
            noise_model=noise_model,
            device=device,
        )
    elif pert_kind == "motion_blur":
        from ddm4ip.degradations.blur import Blur, get_motion_blur_kernel
        filter = get_motion_blur_kernel(
            pert_cfg["kernel_size"],
            pert_cfg["intensity"],
            pert_cfg["rnd_seed"],
        )
        return Blur(
            filter=filter,
            padding=pert_cfg["padding"],
            noise_model=noise_model,
            device = device
        )
    elif pert_kind == "per_pixel_blur":
        from ddm4ip.degradations.varpsf import PerPixelInterpolatedBlur
        return PerPixelInterpolatedBlur(
            psf_path=pert_cfg["psf_path"],
            kernel_size=pert_cfg["kernel_size"],
            noise_model=noise_model,
            device = device
        )
    elif pert_kind == "per_patch_blur":
        from ddm4ip.degradations.varpsf import PerPatchInterpolatedBlur
        return PerPatchInterpolatedBlur(
            psf_path=pert_cfg["psf_path"],
            kernel_size=pert_cfg["kernel_size"],
            padding=pert_cfg["padding"],
            noise_model=noise_model,
            device = device
        )
    elif pert_kind == "none":
        from ddm4ip.degradations.blur import Identity
        return Identity(noise_model=noise_model, 
            device = device)
    else:
        raise ValueError(f"degradation '{pert_kind}' not recognized")


def init_perturbation(pert_cfg, noise_cfg, device):
    noise_model = init_noise(noise_cfg)
    degradation = instantiate_single_kernel(pert_cfg, noise_model, device)
    return degradation


def identity_fn(x, **kwargs):
    return x


def fix_deepinv_state(state):
    """Replace lambdas in the state-dict of deepinv objects with an identity function"""
    for k, v in state.items():
        if callable(v) and v.__name__ == "<lambda>":
            state[k] = identity_fn
            print(f"Replaced lambda at attribute {k} with identity function.")

    return state