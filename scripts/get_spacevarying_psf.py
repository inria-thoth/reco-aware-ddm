import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
import torchvision.transforms.v2.functional as TF
import torchvision


def calc_com(psfs):
    xx, yy = torch.meshgrid(
        torch.arange(psfs.shape[-2], device=psfs.device),
        torch.arange(psfs.shape[-1], device=psfs.device),
        indexing='ij'
    )
    xx = xx[None, None, ...]
    yy = yy[None, None, ...]
    psfs_sum = psfs.sum((-1, -2))
    center_of_mass = (psfs * xx).sum((-1, -2)) / psfs_sum, (psfs * yy).sum((-1, -2)) / psfs_sum
    return center_of_mass


def center_psfs(psfs, com, anchor_channel: int = 0):
    """Centre each PSF so the COM of its `anchor_channel` sits at the grid
    centre. The SAME (scalar) shift is applied to ALL channels, so for an RGB
    PSF the inter-channel (chromatic) displacement is preserved -- only the
    common offset is removed. (Per-channel centring would erase the chromatic
    aberration, which is the whole point of the RGB kernel.)"""
    centered_psfs = []
    for i, psf in enumerate(psfs):
        cy = float(com[0][i, anchor_channel])
        cx = float(com[1][i, anchor_channel])
        centered_psfs.append(
            TF.affine(
                psf,
                translate=[psf.shape[-1] // 2 - cx, psf.shape[-2] // 2 - cy],
                angle=0, scale=1, shear=[0],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                fill=[0.0] * psf.shape[-3],
            )
        )
    return torch.stack(centered_psfs)

def load_mat(base_path: Path):
    mat_path = base_path / "psfs" / "Canon_EF24mm_f_1.4L_USM_ap_1.4.mat"
    if not mat_path.is_file():
        raise ValueError(
            f"Could not find PSF file at '{mat_path.resolve()}'. "
            f"Please download the necessary files manually from https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.4OIMWN "
            f"and unzip them in the directory '{base_path.resolve()}'"
        )
    C = loadmat(str(mat_path))['C']
    N = C.shape[0]
    xs = torch.tensor([C[i, 0][0, 0].astype(int) for i in range(N)])
    ys = torch.tensor([C[i, 1][0, 0].astype(int) for i in range(N)])
    psfs = torch.from_numpy(np.stack([C[i, 3] for i in range(N)]))
    psfs = psfs.permute(0, 3, 1, 2)
    return psfs, xs, ys


def place_psfs_on_grid(xs, ys) -> torch.Tensor:
    ids = []
    pre_transition_ids = None
    is_transition = False
    psf_pos = torch.stack((xs, ys), 1).float()
    all_yy = torch.arange(0, ys.max(), 5).float()
    for x in range(0, xs.max(), 10):
        all_locs = torch.stack((torch.tensor([x], dtype=torch.float32).repeat(len(all_yy)), all_yy), dim=1)
        distances = torch.cdist(all_locs, psf_pos)
        index = torch.argmin(distances, dim=1)
        uc = torch.unique_consecutive(index.reshape(-1))
        if len(ids) == 0:
            ids.append(uc)
            continue
        if set(uc.tolist()) == set(ids[-1].tolist()):
            continue
        if not is_transition:
            is_transition = True
            pre_transition_ids = ids[-1]
        if len(set(uc.tolist()).intersection(set(pre_transition_ids.tolist()))) > 0:
            continue
        else:
            is_transition = False
            ids.append(uc)
    all_ids = torch.stack(ids).T
    return all_ids


def center_psfs_iter(psfs, num_iter: int, anchor_channel: int = 0):
    for _ in range(num_iter):
        com = calc_com(psfs)
        print(f"Error: {(com[0] - psfs.shape[-2] // 2).sum() + (com[1] - psfs.shape[-1] // 2).sum()}")
        psfs = center_psfs(psfs, com, anchor_channel=anchor_channel)
    com = calc_com(psfs)
    print(f"Error: {(com[0] - psfs.shape[-2] // 2).sum() + (com[1] - psfs.shape[-1] // 2).sum()}")
    return psfs


def run(data_path, save_path, channels: str = "green"):
    psfs, xs, ys = load_mat(data_path)
    grid_ids = place_psfs_on_grid(xs, ys)

    m_psfs = TF.to_dtype(psfs, dtype=torch.float32, scale=True)
    if channels == "green":
        m_psfs = m_psfs[:, 1:2]
        anchor = 0
        out_name = "8x8_realpsf_27.pt"
    elif channels == "rgb":
        anchor = 1
        out_name = "8x8_realpsf_rgb_27.pt"
    else:
        raise ValueError(f"--channels must be 'green' or 'rgb', got {channels!r}")
    m_psfs = TF.center_crop(m_psfs, [55, 55])
    m_psfs = TF.resize(m_psfs, [27, 27])
    m_psfs = center_psfs_iter(m_psfs, 3, anchor_channel=anchor)
    m_psfs = m_psfs / m_psfs.sum(dim=(-1, -2), keepdim=True)

    ksize = 27
    num_x = 8
    num_y = 8
    all_x = torch.linspace(0, 1, num_x)
    all_y = torch.linspace(0, 1, num_y)
    psf_im = torch.zeros(m_psfs.shape[-3], num_y * ksize, num_x * ksize)
    out_psfs, out_xs, out_ys = [], [], []
    for i in range(num_y):
        for j in range(num_x):
            grid_id = grid_ids[i * grid_ids.shape[0] // num_y, j * grid_ids.shape[1] // num_x,]
            c_psf = m_psfs[grid_id]
            c_psf = TF.to_dtype(c_psf, dtype=torch.float32, scale=True)
            psf_im[:, i * ksize: (i + 1) * ksize, j * ksize: (j + 1) * ksize] = c_psf
            out_psfs.append(c_psf)
            out_xs.append(all_x[j])
            out_ys.append(all_y[i])
    torch.save({
        "psf": torch.stack(out_psfs, 0),
        "x": torch.tensor(out_xs),
        "y": torch.tensor(out_ys)
    }, str(save_path / out_name))
    print(f"Saved {torch.stack(out_psfs, 0).shape} -> {save_path / out_name}")
    fig, ax = plt.subplots(figsize=(16, 13))
    disp = psf_im if psf_im.shape[0] == 3 else psf_im.repeat(3, 1, 1)
    ax.imshow((disp / disp.max()).permute(1, 2, 0))
    fig.savefig(str(save_path / out_name.replace(".pt", ".png")), dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Processes true PSFs from Bauer et al., https://arxiv.org/pdf/1805.01872 into a simplified but realistic "
        "8x8 space-varying PSF grid. The data can be downloaded from "
        "https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.4OIMWN"
    ))
    parser.add_argument("--data-path", required=True, type=str, help="Path to unzipped original PSF data.")
    parser.add_argument("--save-path", required=True, type=str, help="Directory where to save the .pt file with the processed PSF grid.")
    parser.add_argument("--channels", choices=["green", "rgb"], default="green",
                        help="green = monochrome PSF (8x8_realpsf_27.pt); "
                             "rgb = per-channel chromatic PSF, green-anchored "
                             "(8x8_realpsf_rgb_27.pt).")
    args = parser.parse_args()
    run(Path(args.data_path), Path(args.save_path), channels=args.channels)

"""
python get_spacevarying_psf.py --data-path "$DATA data" --save-path notebooks/
"""