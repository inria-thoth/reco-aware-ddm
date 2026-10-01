#!/usr/bin/env python
"""Project a kernel onto the anisotropic-Gaussian family:

    G(k) = N(mu_k, Sigma_k) on the same grid, renormalised to sum to one,
    where mu_k and Sigma_k are the kernel's centroid and second central moment;
    output = (1-alpha)*k + alpha*G(k), alpha=1 being the full projection.

  python scripts/superresolution/project_kernel_gaussian.py \
      --in  results/<step2run>/kernel_collapsed_kcrop1.mat \
      --out results/<step2run>/kernel_collapsed_kcrop1_gproj1.mat --alpha 1
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def project(k: np.ndarray, alpha: float = 1.0, ridge: float = 1e-3) -> np.ndarray:
    k = np.maximum(np.asarray(k, dtype=np.float64), 0.0)
    s = k.sum()
    if s <= 0:
        raise ValueError("kernel has no positive mass")
    k = k / s
    H, W = k.shape
    ys = np.arange(H)[:, None].astype(np.float64)
    xs = np.arange(W)[None, :].astype(np.float64)
    cy = float((k * ys).sum())
    cx = float((k * xs).sum())
    dy, dx = ys - cy, xs - cx
    syy = float((k * dy * dy).sum()) + ridge
    sxx = float((k * dx * dx).sum()) + ridge
    sxy = float((k * dy * dx).sum())
    det = max(syy * sxx - sxy * sxy, 1e-12)
    quad = (sxx * dy * dy - 2.0 * sxy * dy * dx + syy * dx * dx) / det
    g = np.exp(-0.5 * quad)
    g /= g.sum()
    out = (1.0 - alpha) * k + alpha * g
    return out / out.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="1.0 = full projection, 0 = identity, in between blends")
    a = ap.parse_args()
    m = loadmat(a.src)
    key = "Kernel" if "Kernel" in m else next(
        k for k in m if not k.startswith("__"))
    k = np.asarray(m[key], dtype=np.float64)
    out = project(k, alpha=a.alpha)
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    savemat(a.dst, {"Kernel": out})
    print(f"projected {k.shape} kernel (alpha={a.alpha}) -> {a.dst}   sum {out.sum():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
