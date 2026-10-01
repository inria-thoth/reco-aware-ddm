#!/usr/bin/env python
"""Build clean/degraded zips from the `real_dataset2` capture (79 RW2+JPG shots).

The capture has five sequences, each photographing one scene at both apertures,
with the camera re-oriented and re-centred between shots, so the two apertures
are unpaired:

    seq  frames        f/5.6            f/16
    1    2238-2257     2238-2249 (12)   2250-2257 (8)
    2    2258-2273     2258-2265 (8)    2267-2273 (7)
    3    2275-2288     2275-2281 (7)    2282-2288 (7)
    4    2289-2301     2289-2295 (7)    2296-2301 (6)
    5    2302-2318     2302-2309, 2314-2318 (13)   2310-2313 (4)

The f/16 full frames are the degraded set and the f/5.6 centre crops the clean
set. The RAW->PNG conversion (16 bit) is cached; the selected sequences are
encoded in the output filenames. The paper uses sequence 2:

  python dataset_processing/preprocess_real_dataset2.py \
      --path $DATA/real_dataset2 --preprocess --sequences 2
"""
import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

SEQUENCES = {
    1: (2238, 2257),
    2: (2258, 2273),
    3: (2275, 2288),
    4: (2289, 2301),
    5: (2302, 2318),
}
F_CLEAN = "5.6"
F_NOISY = "16.0"


def frame_number(path: Path) -> int:
    m = re.search(r"P(\d+)", path.stem)
    if not m:
        raise ValueError(f"cannot parse a frame number from {path.name}")
    return int(m.group(1)[-4:])


def sequence_of(num: int):
    for s, (lo, hi) in SEQUENCES.items():
        if lo <= num <= hi:
            return s
    return None


def read_apertures(raw_dir: Path, cache: Path) -> dict:
    """{frame_number: aperture string}, cached (dcraw -i is slow over a network FS)."""
    if cache.is_file():
        try:
            return {int(k): v for k, v in json.load(open(cache)).items()}
        except Exception:
            pass
    out = {}
    raws = sorted(raw_dir.glob("*.RW2"))
    if not raws:
        raise SystemExit(f"no .RW2 files in {raw_dir}")
    for i, raw in enumerate(raws, 1):
        txt = subprocess.run(["dcraw", "-i", "-v", str(raw)],
                             capture_output=True, text=True).stdout
        m = re.search(r"^Aperture: f/([0-9.]+)$", txt, re.MULTILINE)
        if not m:
            print(f"  [warn] no aperture in {raw.name}, skipping")
            continue
        out[frame_number(raw)] = m.group(1)
        if i % 20 == 0:
            print(f"  ...read {i}/{len(raws)} headers")
    json.dump({str(k): v for k, v in out.items()}, open(cache, "w"))
    return out


def raw_to_png(raw_dir: Path, out_base: Path, apertures: dict, suffix: str):
    """dcraw -> ppm -> 16-bit png. Cached per file; inputs are left untouched.

    Flags mirror the (fixed) parking pipeline: 16-bit, hard highlight clip so the
    saturation mask sees a clean plateau, AHD demosaic, median passes against
    demosaic colour zipper, and NO wavelet denoise (it softens the fine detail
    that deblurring would otherwise turn into staircase artefacts).

    `-t 0` DISABLES the EXIF rotation. Without it dcraw applies the camera's
    recorded orientation, which quarter-turns any frame shot with the camera on
    its side -- measured on this capture, 12 of 28 RW2 files, including
    P1082271 in the seq2 f/16 set. That matters because the operator is a grid
    of PSFs indexed in IMAGE coordinates while the aberration field is fixed to
    the SENSOR: a rotated frame transposes the 9x12 grid and mixes field
    positions across frames. Keeping the sensor frame also preserves the
    deliberately inverted captures (a 180-degree camera flip is not recorded in
    EXIF, so those were already passing through untouched).
    """
    ppm_dir = out_base / f"ppm{suffix}"
    png_dir = out_base / f"png{suffix}"
    ppm_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for raw in sorted(raw_dir.glob("*.RW2")):
        num = frame_number(raw)
        ap = apertures.get(num)
        if ap is None:
            continue
        stem = f"{raw.stem}_f{ap}"
        png = png_dir / f"{stem}.png"
        if png.is_file():
            continue
        ppm = ppm_dir / f"{stem}.ppm"
        if not ppm.is_file():
            subprocess.check_output([
                "dcraw", "-v",
                "-6",
                "-w",
                "-H", "0",
                "-q", "3",
                "-m", "3",
                "-t", "0",
                str(raw),
            ])
            produced = raw.with_suffix(".ppm")
            produced.replace(ppm)
        subprocess.check_output([
            "convert", str(ppm),
            "-depth", "16",
            "-define", "png:color-type=2",
            "-quality", "7",
            str(png),
        ])
        print(f"  {raw.name} -> {png.name}")
    return png_dir


def centre_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {img.shape} smaller than crop {size}")
    return img[(h - size) // 2:(h - size) // 2 + size,
               (w - size) // 2:(w - size) // 2 + size]


def build_zip(png_dir: Path, out_zip: Path, frames, aperture: str, crop):
    """Write the selected frames at `aperture` into out_zip, optionally centre-cropped.
    16 bit is preserved: images are re-encoded with cv2 (IMREAD_UNCHANGED)."""
    n, depth = 0, None
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for num in sorted(frames):
            hits = list(png_dir.glob(f"*{num}_f{aperture}.png"))
            if not hits:
                continue
            src = hits[0]
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [skip] unreadable {src.name}")
                continue
            if crop:
                img = centre_crop(img, crop)
            ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                print(f"  [skip] encode failed {src.name}")
                continue
            z.writestr(src.name, buf.tobytes())
            depth = img.dtype
            n += 1
    print(f"  wrote {n:3d} images ({depth}, crop={crop or 'full'}) -> {out_zip.name}")
    return n


def run(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True,
                    help="the real_dataset2 directory (holds the .RW2/.JPG files)")
    ap.add_argument("--sequences", type=int, nargs="+", default=sorted(SEQUENCES),
                    choices=sorted(SEQUENCES),
                    help="which sequences to include (default: all five). "
                         "The selection is encoded in the output file names.")
    ap.add_argument("--preprocess", action="store_true",
                    help="run dcraw/convert to produce the 16-bit PNGs first "
                         "(cached: already-converted frames are skipped)")
    ap.add_argument("--crop", type=int, default=768,
                    help="centre-crop size for the f/5.6 clean images (default 768; "
                         "0 = keep full frames)")
    ap.add_argument("--suffix", default="",
                    help="suffix for the png/ppm dirs and the output zips, so a new "
                         "dataset version does not overwrite the previous one")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the zips (default: --path)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], metavar="FRAME",
                    help="frame numbers to leave out, e.g. --exclude 2271. Use it to "
                         "drop frames whose EXIF orientation made dcraw quarter-turn "
                         "them: the operator is a grid of PSFs indexed in IMAGE "
                         "coordinates while the aberration field is fixed to the "
                         "SENSOR, so a rotated frame transposes the grid.")
    ap.add_argument("--tag", default=None,
                    help="override the tag used in the output names (default: derived "
                         "from --sequences, e.g. 'seq2'). Give a distinct tag whenever "
                         "--exclude changes the contents, e.g. --tag seq2b.")
    args = ap.parse_args(argv[1:])

    base = Path(args.path)
    if not base.is_dir():
        raise SystemExit(f"no such directory: {base}")
    out_dir = Path(args.out_dir) if args.out_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading apertures from the RAW headers...")
    apertures = read_apertures(base, base / f".apertures{args.suffix}.json")

    png_dir = base / f"png{args.suffix}"
    if args.preprocess:
        print("Converting RAW -> 16-bit PNG (cached)...")
        png_dir = raw_to_png(base, base, apertures, args.suffix)
    elif not png_dir.is_dir():
        raise SystemExit(f"{png_dir} does not exist -- run once with --preprocess")

    sel = sorted(set(args.sequences))
    excluded = set(args.exclude)
    frames = {ap_: [] for ap_ in (F_CLEAN, F_NOISY)}
    dropped = []
    for num, ap_ in apertures.items():
        if sequence_of(num) in sel and ap_ in frames:
            if num in excluded:
                dropped.append((num, ap_))
                continue
            frames[ap_].append(num)
    if excluded:
        for num, ap_ in sorted(dropped):
            print(f"  excluded frame {num} (f/{ap_})")
        missing = excluded - {n for n, _ in dropped}
        if missing:
            print(f"  [warn] --exclude listed frames not in the selection: "
                  f"{sorted(missing)}")

    tag = args.tag or ("all" if sel == sorted(SEQUENCES)
                       else "seq" + "".join(map(str, sel)))
    if excluded and not args.tag:
        raise SystemExit(
            "--exclude changes the dataset contents but the tag is unchanged, so "
            "the zips would overwrite the full-sequence ones. Pass --tag (e.g. "
            "--tag seq2b).")
    print(f"\nSequences {sel} -> tag '{tag}': "
          f"{len(frames[F_CLEAN])} f/{F_CLEAN}, {len(frames[F_NOISY])} f/{F_NOISY}")

    crop = args.crop if args.crop > 0 else None
    clean_name = (f"real2_{tag}_5_6_center{crop}{args.suffix}.zip" if crop
                  else f"real2_{tag}_5_6{args.suffix}.zip")
    build_zip(png_dir, out_dir / clean_name, frames[F_CLEAN], F_CLEAN, crop)
    build_zip(png_dir, out_dir / f"real2_{tag}_16{args.suffix}.zip",
              frames[F_NOISY], F_NOISY, None)

    print(f"\nClean (x): {out_dir / clean_name}")
    print(f"Noisy (y): {out_dir / ('real2_%s_16%s.zip' % (tag, args.suffix))}")


if __name__ == "__main__":
    run(sys.argv)
