import io
import sys
import zipfile
from pathlib import Path
import numpy as np
import tqdm
from PIL import Image


def _select(src: Path, ids) -> list:
    """Map FFHQ indices to the 1024x1024 source images, in the order given."""
    by_id = {int(q.stem): q for q in src.rglob("*.png") if q.stem.isdigit()}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} FFHQ images not found under {src}, e.g. {missing[:5]}")
    return [by_id[i] for i in ids]


def _write(images, out_file: Path, size: int = 256):
    if out_file.exists():
        print(f"{out_file} exists, skipping.")
        return
    with zipfile.ZipFile(out_file, mode="w", compression=0) as zf:
        for q in tqdm.tqdm(images, desc=out_file.name):
            with Image.open(q) as im:
                if im.size != (1024, 1024):
                    raise ValueError(f"{q}: expected 1024x1024, found {im.size}")
                buf = io.BytesIO()
                im.resize((size, size)).save(buf, format="PNG")
            zf.writestr(q.name, buf.getvalue())


def run(argv):
    import argparse
    parser = argparse.ArgumentParser(
        argv[0], description="Build the FFHQ-256 subsets of the uniform-blur experiment.")
    parser.add_argument("--input-path", required=True,
                        help="Directory with the 70,000 FFHQ 1024x1024 PNGs (any sub-folder layout).")
    parser.add_argument("--output-path", required=True, help="Where to write the FFHQ-256 zip files.")
    args = parser.parse_args(argv[1:])
    src, out = Path(args.input_path), Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    perm = np.random.RandomState(seed=42).permutation(10_000)
    for offset, tag in ((0, "clean"), (60_000, "noisy")):
        _write(_select(src, (offset + perm[0:1000]).tolist()), out / f"ffhq_256_tr_1k_{tag}.zip")
    _write(_select(src, perm[2000:3000].tolist()), out / "ffhq_256_val_clean.zip")

    ids_file = Path(__file__).with_name("ffhq_256_val_ids.txt")
    ids = [int(l) for l in ids_file.read_text().split("\n") if l.strip() and not l.startswith("#")]
    _write(_select(src, ids), out / "ffhq_256_val.zip")


if __name__ == "__main__":
    run(sys.argv)
