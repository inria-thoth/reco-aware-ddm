from ddm4ip.networks.blur_nets import DegradationNetwork, DirectCenterKernel2, DirectKernel2
from ddm4ip.networks.noise_nets import GaussianNoiseNet


def init_kernel_net(cfg) -> DegradationNetwork:
    """The learnable forward operator (config group `models.kernel`), followed by
    Gaussian noise: a single kernel (`type: direct`, with decimation when
    `factor > 1`) or a grid of kernels interpolated at the patch centre
    (`type: direct_center`)."""
    if cfg['noise_type'] != "gaussian":
        raise ValueError(f"models.kernel.noise_type '{cfg['noise_type']}' is not supported (use 'gaussian').")
    noise_nn = GaussianNoiseNet(fixed_sigma=cfg.get("output_noise_sigma", None))

    kernel_type = cfg['type']
    if kernel_type == "direct":
        kernel_nn = DirectKernel2(
            kernel_size=cfg['kernel_size'],
            padding=cfg.get('padding', 'replicate'),
            factor=cfg.get('factor', 1),
            kernel_norm=cfg.get("kernel_norm", None),
            kernel_ch=cfg.get("kernel_ch", 1),
            auto_center=cfg.get("auto_center", False),
            auto_center_window=cfg.get("auto_center_window", 8.0),
            auto_center_anchor_channel=cfg.get("auto_center_anchor_channel", 0),
            initialization=cfg.get("initialization", "random"),
        )
    elif kernel_type == "direct_center":
        kernel_nn = DirectCenterKernel2(
            psf_path=cfg["psf_path"],
            kernel_size=cfg["kernel_size"],
            padding=cfg["padding"],
            num_psfs=cfg.get("num_psfs", None),
            kernel_ch=cfg.get("kernel_ch", None),
            initialization=cfg["initialization"],
            kernel_norm=cfg.get("kernel_norm", None),
            auto_center=cfg.get("auto_center", False),
            auto_center_window=cfg.get("auto_center_window", 8.0),
            auto_center_anchor_channel=cfg.get("auto_center_anchor_channel", 0),
        )
    else:
        raise ValueError(f"kernel type (config['models']['kernel']['type']) '{kernel_type}' is not valid.")

    return DegradationNetwork(kernel_nn, noise_nn)
