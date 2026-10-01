#!/usr/bin/env python
"""Kernel MAE of the super-resolution experiment, after sub-pixel centre-of-mass
alignment on a common canvas (numpy only; .mat files are read with a small MAT v5
parser).
"""
import os
import argparse, struct, zlib
from pathlib import Path
import numpy as np

DATA_ROOT = os.environ.get("DDM4IP_DATA", "./data")

_NUMPY_OF = {1:'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4',
             9:'f8', 12:'i8', 13:'u8'}

def _tag(b, o):
    """(dtype, nbytes, data_offset, next_offset) handling the small-element form."""
    w1, w2 = struct.unpack('<II', b[o:o+8])
    if w1 >> 16:
        return w1 & 0xFFFF, w1 >> 16, o + 4, o + 8
    n = w2
    return w1, n, o + 8, o + 8 + (n + 7) // 8 * 8

def _matrix(b, o):
    dt, nb, do, _ = _tag(b, o)
    assert dt == 14, f"not a miMATRIX (dt={dt})"
    p = do
    dt, nb, do, p = _tag(b, p)
    dt, nb, do, p = _tag(b, p)
    dims = struct.unpack('<%di' % (nb // 4), b[do:do+nb])
    dt, nb, do, p = _tag(b, p)
    dt, nb, do, p = _tag(b, p)
    arr = np.frombuffer(b[do:do+nb], dtype='<' + _NUMPY_OF[dt])
    return np.asarray(arr, dtype=np.float64).reshape(dims, order='F')

def loadmat_one(path):
    b = open(path, 'rb').read()
    o = 128
    while o < len(b) - 8:
        dt, nb, do, nxt = _tag(b, o)
        if dt == 15:
            return _matrix(zlib.decompress(b[do:do+nb]), 0)
        if dt == 14:
            return _matrix(b, o)
        o = nxt
    raise ValueError(f"no matrix in {path}")

def to_canvas(k, c):
    h, w = k.shape
    if h > c or w > c:
        t, l = (h - c) // 2, (w - c) // 2
        k = k[t:t+c, l:l+c]; h, w = k.shape
    out = np.zeros((c, c), dtype=k.dtype)
    t, l = (c - h) // 2, (c - w) // 2
    out[t:t+h, l:l+w] = k
    return out

def com(k, eps=1e-6):
    """kernel_com: centroid of |k| in pixel-index coords, eps on the denominator."""
    ka = np.abs(k); H, W = ka.shape
    ys = np.arange(H, dtype=k.dtype)[:, None]
    xs = np.arange(W, dtype=k.dtype)[None, :]
    s = ka.sum() + eps
    return (ka * ys).sum() / s, (ka * xs).sum() / s

def bilinear_shift(k, dy, dx):
    """grid_sample(mode=bilinear, padding=zeros, align_corners=True): out[y]=k[y-dy]."""
    H, W = k.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(k.dtype)
    sy, sx = yy - dy, xx - dx
    y0, x0 = np.floor(sy), np.floor(sx)
    wy, wx = sy - y0, sx - x0
    out = np.zeros_like(k)
    for oy, ox, wt in ((0,0,(1-wy)*(1-wx)), (0,1,(1-wy)*wx),
                       (1,0,wy*(1-wx)),     (1,1,wy*wx)):
        yi = (y0+oy).astype(np.int64); xi = (x0+ox).astype(np.int64)
        m = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
        out[m] += wt[m] * k[yi[m], xi[m]]
    return out

def aligned_mae(pred, true):
    py, px = com(pred); ty, tx = com(true)
    return np.abs(true - bilinear_shift(pred, ty - py, tx - px)).mean()

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", required=True, nargs="+")
ap.add_argument("--label", nargs="*", default=None, help="display names, one per pattern")
ap.add_argument("--gt", default=DATA_ROOT + "/div2krk/gt_k_x2/kernel_{n}.mat")
ap.add_argument("--start", type=int, default=1)
ap.add_argument("--end", type=int, default=100)
ap.add_argument("--canvas", type=int, default=33)
ap.add_argument("--crop", type=int, default=0)
a = ap.parse_args()
labels = a.label if a.label and len(a.label) == len(a.pattern) else a.pattern

print(f"canvas {a.canvas}   images {a.start}..{a.end}   "
      f"crop {a.crop or 'none'}   (clamp>=0 + renormalise, as DPIR uses them)\n")
print(f"{'method':26s} {'n':>3s} {'shape':>7s} {'alignedMAE':>11s} {'rawMAE':>9s} {'COMoff':>7s}")
print("-" * 70)
for lab, pattern in zip(labels, a.pattern):
    A, R, C, shapes = [], [], [], set()
    for n in range(a.start, a.end + 1):
        pe, pg = Path(pattern.format(n=n)), Path(a.gt.format(n=n))
        if not pe.is_file() or not pg.is_file():
            continue
        ke = loadmat_one(pe).astype(np.float32)
        kg = loadmat_one(pg).astype(np.float32)
        shapes.add(ke.shape)
        if a.crop > 0 and (ke.shape[0] > a.crop or ke.shape[1] > a.crop):
            t, l = (ke.shape[0]-a.crop)//2, (ke.shape[1]-a.crop)//2
            ke = ke[t:t+a.crop, l:l+a.crop]
        ke = np.maximum(ke, 0); kg = np.maximum(kg, 0)
        ke = ke / max(ke.sum(), 1e-12); kg = kg / max(kg.sum(), 1e-12)
        ke, kg = to_canvas(ke, a.canvas), to_canvas(kg, a.canvas)
        A.append(aligned_mae(ke, kg) * 1e4)
        R.append(np.abs(ke - kg).mean() * 1e4)
        cy, cx = com(ke)
        C.append(float(np.hypot(cy - (a.canvas-1)/2, cx - (a.canvas-1)/2)))
    if not A:
        print(f"{lab:26s}   no files"); continue
    sh = "/".join(f"{h}x{w}" for h, w in sorted(shapes))
    print(f"{lab:26s} {len(A):3d} {sh:>7s} {np.mean(A):11.2f} {np.mean(R):9.2f} {np.mean(C):7.3f}")
