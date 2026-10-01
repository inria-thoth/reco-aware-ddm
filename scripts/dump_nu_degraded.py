#!/usr/bin/env python
"""Dump the degraded inputs (and clean targets) of the space-varying `nu` experiment.

External baselines that restore an image directly need the degraded images as FILES. The evaluation synthesises them on the fly, so
this script materialises exactly the same pair for every test image:

    y = PerPixelInterpolatedBlur(psf_grid)(x) + sigma * eps,   x = DPDD test target

by building the SAME dataset object the evaluation uses, from the SAME hydra
config (`paper/step3/y_cur/step3_ffhq256_nu_diy`). Nothing about the
degradation is re-implemented here, so the baselines see byte-identical inputs
to what our method is scored on.

The noise is drawn with a fixed seed (--seed, default the config's training.seed)
so the dump is reproducible and the same across baselines.

  python scripts/dump_nu_degraded.py --out $DATA/nu_baseline_inputs
  python scripts/dump_nu_degraded.py --out /tmp/nu --limit 4      # quick check

Writes <out>/degraded/*.png and <out>/clean/*.png, 8-bit sRGB, plus a manifest.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ddm4ip"))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True, help="output directory")
ap.add_argument("--exp", default="paper/step3/y_cur/step3_ffhq256_nu_diy",
                help="hydra exp config to take the test dataset from")
ap.add_argument("--paths", default="default", help="hydra paths group")
ap.add_argument("--data-root", default=None,
                help="override paths.data (the dataset root). Useful when the paths "
                     "group points at a mount this machine does not have.")
ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
ap.add_argument("--seed", type=int, default=None, help="noise seed (default: config training.seed)")
ap.add_argument("--device", default="cuda", help="device for the degradation (default cuda)")
ap.add_argument("--float", action="store_true",
                help="also write float32 .npy next to each PNG, for BOTH the degraded "
                     "and the clean image (no 8-bit rounding, no clipping of the "
                     "pixels that noise pushes below 0). Baselines that read the .npy "
                     "then see exactly what our evaluation feeds the solver.")
args = ap.parse_args()

import hydra
from omegaconf import OmegaConf
from ddm4ip.main import parse_nimg

try:
    OmegaConf.register_new_resolver("parse_nimg", parse_nimg)
except ValueError:
    pass

with hydra.initialize_config_dir(config_dir=str(REPO / "ddm4ip" / "configs"),
                                 version_base=None):
    ov = [f"exp={args.exp}", f"paths={args.paths}"]
    if args.data_root:
        ov.append(f"paths.data={args.data_root}")
    cfg = hydra.compose(config_name="main.yaml", overrides=ov)
cfg = OmegaConf.to_container(cfg, resolve=True)

cfg["dataset"]["test"]["device"] = args.device
seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)
torch.manual_seed(seed)
np.random.seed(seed)

from ddm4ip.data.base import Datasplit, init_dataset

dset = init_dataset(cfg, Datasplit.TEST, is_paired=True)
n = len(dset) if args.limit in (0, None) else min(args.limit, len(dset))
print(f"exp        : {args.exp}")
print(f"test set   : {cfg['dataset']['test'].get('path')}")
print(f"degradation: {cfg['dataset']['test']['degradation']}")
print(f"noise      : {cfg['dataset']['test']['noise']}   seed={seed}")
print(f"images     : {n} (of {len(dset)})")

OUT = Path(args.out)
(OUT / "degraded").mkdir(parents=True, exist_ok=True)
(OUT / "clean").mkdir(parents=True, exist_ok=True)


def to_png(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) -> HxWxC uint8.

    The pipeline works in [0,1] (`v2.ToDtype(torch.float32, scale=True)` in
    data2.init_base_dset), which is also what ddm4ip.utils.metrics.calc_psnr
    assumes. Additive noise can push a few pixels slightly outside, so clamp --
    do NOT rescale, or the range convention would silently change.
    """
    a = t.detach().float().cpu()
    lo, hi = float(a.min()), float(a.max())
    if lo < -0.05 or hi > 1.05:
        print(f"    [warn] values outside [0,1]: min {lo:.3f} max {hi:.3f}")
    a = a.clamp(0, 1).permute(1, 2, 0).numpy()
    return (a * 255.0).round().astype(np.uint8)


from PIL import Image

def needs_float(t: torch.Tensor) -> bool:
    """True when 8-bit PNG cannot represent this tensor exactly.

    The DPDD targets are 16-bit PNGs (`_load_rgb_any_depth` keeps them as uint16
    and divides by 65535), so the clean images carry ~8 bits more than a PNG
    dump can hold, and the degraded ones are float anyway."""
    a = (t.detach().float().cpu() * 255.0)
    return bool((a - a.round()).abs().max() > 1e-3)


warned = False
manifest = []
for i in range(n):
    b = dset[i]
    y = b.corrupt if b.corrupt is not None else None
    x = b.clean
    if y is None:
        raise SystemExit("dataset produced no corrupt image -- wrong config?")
    y = y[0] if y.dim() == 4 else y
    x = x[0] if x is not None and x.dim() == 4 else x
    if i == 0 and not args.float and (needs_float(x) if x is not None else needs_float(y)):
        print("  [warn] the source images carry more than 8 bits (the DPDD PNGs are "
              "16-bit), so the PNG dump is LOSSY: ~59 dB of rounding, plus clipping "
              "of the pixels noise pushes below 0.\n"
              "         Re-run with --float to also write the exact float32 .npy; the "
              "baseline runners and score_nu_baseline.py prefer it automatically.")
        warned = True
    name = f"{i:04d}.png"
    Image.fromarray(to_png(y)).save(OUT / "degraded" / name)
    if x is not None:
        Image.fromarray(to_png(x)).save(OUT / "clean" / name)
    if args.float:
        np.save(OUT / "degraded" / f"{i:04d}.npy", y.detach().float().cpu().numpy())
        if x is not None:
            np.save(OUT / "clean" / f"{i:04d}.npy", x.detach().float().cpu().numpy())
    manifest.append({"index": i, "file": name,
                     "shape": list(y.shape), "clean": x is not None})
    print(f"  {name}  degraded {tuple(y.shape)}"
          + ("" if x is None else f"  clean {tuple(x.shape)}"))

(OUT / "manifest.json").write_text(json.dumps(
    {"exp": args.exp, "seed": seed, "n": n,
     "degradation": cfg["dataset"]["test"]["degradation"],
     "noise": cfg["dataset"]["test"]["noise"],
     "images": manifest}, indent=2))
print(f"\nwrote {n} pairs to {OUT}")
if warned:
    print("  [warn] 8-bit PNGs only -- see the warning above; --float writes the "
          "exact tensors alongside them")
print(f"  {OUT/'degraded'}  <- feed this to the baselines")
print(f"  {OUT/'clean'}     <- ground truth for scoring")
print(f"  {OUT/'manifest.json'}")
