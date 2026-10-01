import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib
matplotlib.use("agg", force=True)

from ddm4ip.trainers.deepinv_denoiser import DeepinvDenoiserTrainer
from ddm4ip.trainers.diffinstruct_learn_op import DiffinstructOpTrainer
from ddm4ip.trainers.finetune_flow import FinetuneFlowTrainer
from ddm4ip.trainers.di_learn_2_op import DI2OpTrainer
from ddm4ip.utils import distributed


def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

OmegaConf.register_new_resolver("parse_nimg", parse_nimg)

@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def my_app(cfg : DictConfig) -> None:
    # NOTE:
    # Running distributed.init seems to cause problems with dataset pickling
    # (pickling the lambdas in deepinv). It's unclear what the problem could
    # be (maybe just removing the `set_start_method` could be enough). This
    # has been observed on a cluster even when running on a single GPU, so has been
    # disabled.

    torch.multiprocessing.set_start_method('forkserver')
    try:
        distributed.init()

        # TF32 for float32 matrix multiplications (training.tf32).
        if cfg.training.get("tf32", False):
            torch.set_float32_matmul_precision("high")

        if "deepinv_solver" in cfg.models:
            tr_settings = DeepinvDenoiserTrainer()
        elif 'di_x_y' in cfg['loss']['name']:
            tr_settings = DI2OpTrainer()
        elif "pretrained_flow" in cfg.models and "kernel" in cfg.models:
            tr_settings = DiffinstructOpTrainer()
        elif "flow" in cfg.models:
            tr_settings = FinetuneFlowTrainer()
        else:
            raise RuntimeError(
                f"Config 'models' contains keys {list(cfg.models.keys())} "
                "which do not correspond to a known trainer."
            )
        if cfg.training.train is True:
            tr_settings.train(cfg)
        elif cfg.training.train is False:
            tr_settings.evaluate(cfg)
        else:
            raise RuntimeError(
                "Neither 'training' nor 'validation' configuration found."
            )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    my_app()