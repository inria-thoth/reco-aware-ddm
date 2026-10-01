"""Zero a frame of `--border` pixels around a kernel, recenter it (sub-pixel,
bilinear, zero padding, as in ddm4ip.utils.torch_utils.center_kernel_via_shift)
and renormalize it to sum to one.

  python scripts/superresolution/crop_kernel_border.py \
      --in  results/<step2run>/kernel_collapsed.mat \
      --out results/<step2run>/kernel_collapsed_kcrop1.mat --border 1
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def com(k, eps=1e-8):
    ka = np.abs(k)
    H, W = ka.shape
    ys = np.arange(H, dtype=np.float64)[:, None]
    xs = np.arange(W, dtype=np.float64)[None, :]
    s = ka.sum() + eps
    return float((ka * ys).sum() / s), float((ka * xs).sum() / s)


def bilinear_shift(k, dy, dx):
    """out[y,x] = k[y-dy, x-dx], bilinear, zeros outside."""
    H, W = k.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    sy, sx = yy - dy, xx - dx
    y0, x0 = np.floor(sy), np.floor(sx)
    wy, wx = sy - y0, sx - x0
    out = np.zeros_like(k)
    for oy, ox, wt in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx),
                       (1, 0, wy * (1 - wx)),       (1, 1, wy * wx)):
        yi = (y0 + oy).astype(np.int64)
        xi = (x0 + ox).astype(np.int64)
        m = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
        out[m] += wt[m] * k[yi[m], xi[m]]
    return out


def crop_border(k, border=1):
    k = np.maximum(np.asarray(k, dtype=np.float64), 0.0)
    if k.sum() <= 0:
        raise ValueError("kernel has no positive mass")
    b = int(border)
    if b > 0:
        H, W = k.shape
        if 2 * b >= min(H, W):
            raise ValueError(f"border={b} removes the whole {H}x{W} kernel")
        k = k.copy()
        k[:b, :] = 0.0
        k[H - b:, :] = 0.0
        k[:, :b] = 0.0
        k[:, W - b:] = 0.0
    if k.sum() <= 0:
        raise ValueError("zeroing removed all the mass")
    H, W = k.shape
    cy, cx = com(k)
    k = bilinear_shift(k, (H - 1) / 2.0 - cy, (W - 1) / 2.0 - cx)
    k = np.maximum(k, 0.0)
    return k / k.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--border", type=int, default=1, help="pixels to zero on each side")
    a = ap.parse_args()
    m = loadmat(a.src)
    key = "Kernel" if "Kernel" in m else next(k for k in m if not k.startswith("__"))
    k = np.asarray(m[key], dtype=np.float64)
    out = crop_border(k, border=a.border)
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    savemat(a.dst, {"Kernel": out})
    print(f"border={a.border} zeroed: {k.shape} -> {a.dst}   sum {out.sum():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
