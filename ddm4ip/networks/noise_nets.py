import deepinv
import torch
import torch.nn as nn

from ddm4ip.utils import persistence


@persistence.persistent_class
class GaussianNoiseNet(nn.Module):
    """Additive Gaussian noise on the simulated measurements. The noise level is
    the one supplied with the batch (the dataset noise level) or, when the
    dataset supplies none, the constant `fixed_sigma`."""

    def __init__(self, fixed_sigma: float | None = None):
        super().__init__()
        # A buffer, so that it follows .to(device) and is saved with the model.
        buf = None
        if fixed_sigma is not None:
            buf = torch.tensor([float(fixed_sigma)])
        self.register_buffer("fixed_sigma", buf)

    def get_noisemodel(self, **noise_kwargs):
        noise_model = deepinv.physics.GaussianNoise()
        for pname, p in self.get_parameters(**noise_kwargs).items():
            setattr(noise_model, pname, p)
        return noise_model

    def get_parameters(self, sigma: float | torch.Tensor | None = None, **kwargs) -> dict[str, torch.Tensor]:
        if sigma is None:
            if self.fixed_sigma is not None:
                return {"sigma": self.fixed_sigma}
            raise ValueError("Noise-level was not specified and no fixed noise level is set.")
        if isinstance(sigma, (float, int)):
            # deepinv does not move a float noise level to the device.
            sigma = torch.tensor(sigma, device=torch.cuda.current_device(), dtype=torch.float32)
        return {"sigma": sigma}
