#!/usr/bin/env python
"""Score an external baseline on the `nu` experiment with OUR metric code.

The reported nu numbers are FULL-IMAGE metrics: `DeepInvLoss.run_model` splits
the image into patches only to bound memory, stitches them back with
`patches2img`, and `compute_img_metrics` then scores the reassembled image. So a
baseline that restores whole images is directly comparable -- provided the same
metric functions and conventions are used, which is what this script does: it
calls `ddm4ip.utils.metrics` (calc_psnr / calc_ssim / calc_psnr_matlab) and the
same LPIPS module, with `eval_crop_border` (0 for nu; the SR config uses 2).

Inputs are two directories of matching filenames, as produced by
scripts/dump_nu_degraded.py:

    <restored>/0000.png ...     the baseline's output
    <clean>/0000.png ...        the ground truth

  python scripts/score_nu_baseline.py \\
      --restored $DATA/nu_baseline_inputs/omnilens_out \\
      --clean    $DATA/nu_baseline_inputs/clean --label OmniLens
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ddm4ip"))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--restored", required=True, help="directory of baseline outputs")
ap.add_argument("--clean", required=True, help="directory of ground-truth images")
ap.add_argument("--label", default="baseline", help="name for the printed row")
ap.add_argument("--crop-border", type=int, default=0,
                help="pixels shaved from each side before scoring "
                     "(0 = the nu setting; the SR config uses 2)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--inputs", choices=("auto", "npy", "png"), default="auto",
                help="'auto' (default) reads <stem>.npy where it exists, so a "
                     "baseline that kept its float output is scored without the "
                     "8-bit round trip; 'png' forces the quantized copy")
ap.add_argument("--csv", default=None, help="also write per-image rows here")
args = ap.parse_args()

from PIL import Image
import torchvision.transforms.functional as TF
from ddm4ip.utils.metrics import calc_psnr, calc_psnr_matlab, calc_ssim

from ddm4ip.utils.metrics import LPIPS
lpips_fn = LPIPS(net_type="alex").to(args.device)


IMG = (".png", ".jpg", ".jpeg", ".bmp")


def load(p: Path) -> torch.Tensor:
    """-> (1,3,H,W) float in [0,1].

    That is the pipeline's convention (`ToDtype(..., scale=True)`) and what
    calc_psnr/calc_ssim assume; feeding [-1,1] would misreport by ~6 dB. A .npy
    is taken as-is (float32, CHW), a PNG is divided by 255.
    """
    if p.suffix == ".npy":
        a = np.load(p).astype(np.float32)
        if a.ndim == 3 and a.shape[0] not in (1, 3):
            a = a.transpose(2, 0, 1)
        return torch.from_numpy(a)[None]
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


SUFFIXES = ("_fake_B", "_fake_b", "_fake", "_out", "_output", "_restored", "_pred")


def _stem(p: Path) -> str:
    s = p.stem
    for suf in SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def index(d: Path):
    """{stem: path}, float .npy preferred over an 8-bit copy of the same frame."""
    npy = {_stem(p): p for p in d.glob("*.npy")}
    png = {_stem(p): p for p in d.iterdir() if p.suffix.lower() in IMG}
    if args.inputs == "npy":
        return npy
    if args.inputs == "png":
        return png
    return {**png, **npy}


rest_dir, clean_dir = Path(args.restored), Path(args.clean)
REST, CLEAN = index(rest_dir), index(clean_dir)
names = sorted(CLEAN)
if not names:
    raise SystemExit(f"no images in {clean_dir}")
nf = sum(1 for s in names if REST.get(s) is not None and REST[s].suffix == ".npy")
ng = sum(1 for s in names if CLEAN[s].suffix == ".npy")
print(f"scoring {len(names)} frames; restorations read as "
      f"{nf} float .npy / {len(names) - nf} 8-bit; "
      f"ground truth {ng} float .npy / {len(names) - ng} 8-bit")
if ng == 0 and args.inputs != "png":
    print("  [warn] the ground truth is 8-bit, but the DPDD targets are 16-bit PNGs: "
          "this rounds the reference by ~59 dB (about 0.01 dB of PSNR here).\n"
          "         Re-run scripts/dump_nu_degraded.py with --float to write clean/*.npy.")

rows, missing = [], []
for name in names:
    rp = REST.get(name)
    if rp is None:
        missing.append(name)
        continue
    rc, gt = load(rp).to(args.device), load(CLEAN[name]).to(args.device)
    if rc.shape != gt.shape:
        raise SystemExit(f"{name}: restored {tuple(rc.shape)} != clean {tuple(gt.shape)}; "
                         "the baseline must not crop or resize")
    cb = args.crop_border
    if cb > 0:
        rc = TF.center_crop(rc, [rc.shape[-2] - 2 * cb, rc.shape[-1] - 2 * cb])
        gt = TF.center_crop(gt, [gt.shape[-2] - 2 * cb, gt.shape[-1] - 2 * cb])
    with torch.no_grad():
        m = {
            "psnr": float(calc_psnr(gt, rc)),
            "psnr_matlab": float(calc_psnr_matlab(gt, rc)),
            "ssim": float(calc_ssim(gt, rc)),
            "lpips": float(lpips_fn(gt, rc).mean()),
        }
    rows.append((name, m))
    print(f"  {name}  psnr {m['psnr']:6.2f}  psnr-Y {m['psnr_matlab']:6.2f}  "
          f"ssim {m['ssim']:.4f}  lpips {m['lpips']:.4f}")

if missing:
    print(f"\n[warn] {len(missing)} images had no restoration: {missing[:5]}"
          + (" ..." if len(missing) > 5 else ""))
if not rows:
    raise SystemExit("nothing scored")

mean = {k: float(np.mean([m[k] for _, m in rows])) for k in rows[0][1]}
print(f"\n{args.label}: n={len(rows)}  PSNR {mean['psnr']:.3f}  "
      f"PSNR-Y {mean['psnr_matlab']:.3f}  SSIM {mean['ssim']:.4f}  LPIPS {mean['lpips']:.4f}"
      f"   (crop_border={args.crop_border})")
print("reference  ours 32.410 / 0.925 / 0.132 | meas.-only 23.853 / 0.838 / 0.207 "
      "| true operator 35.523 / 0.939 / 0.111   (PSNR / SSIM / LPIPS)")

if args.csv:
    with open(args.csv, "w") as fh:
        fh.write("image,psnr,psnr_matlab,ssim,lpips\n")
        for name, m in rows:
            fh.write(f"{name},{m['psnr']:.4f},{m['psnr_matlab']:.4f},"
                     f"{m['ssim']:.4f},{m['lpips']:.4f}\n")
    print("wrote", args.csv)
