# Reconstruction-Aware Distribution Matching for Inverse Problems with Unpaired Data

Code to reproduce the experiments of the paper "Reconstruction-Aware Distribution Matching
for Inverse Problems with Unpaired Data"

Given a set of clean images and a separate, unpaired set of degraded images, the
method learns an explicit forward operator `A`. The measurement term matches the
distribution of simulated measurements `A(x) + noise` with the degraded set; the
reconstruction term matches the distribution of the degraded images restored
with `A` (by a differentiable DPIR solver) with the clean set. Both terms are
estimated with flow-matching teachers and auxiliary fields, and their gradients
are combined by gradient-norm balancing with a linear warm-up of the weight
`lambda`. The measurement-only baseline is the same code without the
reconstruction term.

## Authors

the public implementation of the paper Unsupervised Imaging Inverse Problems
with Diffusion Distribution Matching (Meanti, Ryckeboer, Arbel and Mairal,
2025). The first version of this package was written by Thomas Ryckeboer and
Giacomo Meanti while they were at Inria. Julien Mairal then took over the
project and developed it, using Claude Code as a coding assistant.

## Layout

```
ddm4ip/
  main.py                   entry point (hydra); selects the stage from the config
  configs/                  experiment, dataset, model, loss and path configurations
  losses/di_y.py            measurement-only objective (baseline)
  losses/di_x_y.py          measurement + reconstruction objective (ours)
  losses/flow_matching.py   teachers (stage 1) and auxiliary fields
  losses/deepinv_loss.py    stage 3: restoration and metrics
  networks/blur_nets.py     operator parameterisations and centering
  networks/unets.py         velocity fields
  pnp/dpir.py               differentiable DPIR
  degradations/, psf/       synthetic degradations and the measured PSF field
  data/, trainers/, utils/
scripts/
  uniform_clean_teacher.sh  uniform blur: clean teacher, shared by the nine kernels
  uniform_<kernel>.sh       uniform blur: one script per kernel
  psf_field.sh              space-varying PSF field
  super_resolution.sh       blind super-resolution, one image
  super_resolution_all.sh   blind super-resolution, a range of images
  real_capture.sh           real unpaired capture
  superresolution/          kernel export and post-processing (super-resolution)
  sr_kernel_mae_np.py       super-resolution kernel error on a common canvas
  get_spacevarying_psf.py   builds the measured PSF grid
  dump_nu_degraded.py       writes the space-varying test pairs to files
  score_nu_baseline.py      scores restorations of those pairs with the same metrics
dataset_processing/         dataset preparation
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Every run fits on one GPU. The scripts call `python`; set
`DDM4IP_PYTHON_BIN=/path/to/venv/bin` to use a specific interpreter.

## Paths

Datasets are read from `paths.data` (`ddm4ip/configs/paths/default.yaml`,
default `./data`). The scripts write every run to `results/<run name>/` in the
repository root (checkpoints, tensorboard logs and `metrics.jsonl`). Another
paths file can be selected with `PATHS=<name>`.

## Data

| Experiment | Clean set | Degraded set | Evaluation |
|---|---|---|---|
| Uniform blur | FFHQ-256, 1k images | FFHQ-256, 1k other images + synthetic kernel | FFHQ-256, 1k images |
| Space-varying PSF | DPDD, sharp | DIV2K + measured PSF field | DPDD test, 76 images |
| Blind super-resolution | the LR image | crops of the LR image | DIV2KRK, 100 images |
| Real capture | f/5.6 centre crops | f/16 frames | qualitative |

```bash
python dataset_processing/preprocess_ffhq.py \
    --input-path /path/to/ffhq/images1024x1024 --output-path data/ffhq
python dataset_processing/preprocess_div2k.py --path data/div2k
unzip dd_dp_dataset_canon.zip -d data
python scripts/get_spacevarying_psf.py --data-path data/psf_raw \
    --save-path data/psfs --channels rgb
python dataset_processing/preprocess_real_dataset2.py \
    --path data/real_dataset2 --preprocess --sequences 2
```

DPDD is read from `data/dd_dp_dataset_png/{train_c,test_c}/target`. The PSF
field is the calibration data of Bauer et al. (2018), doi:10.17617/3.4OIMWN.
DIV2KRK (KernelGAN release) is expected in `data/div2krk/{gt,lr_x2,gt_k_x2}`.
The FFHQ evaluation set is given by `dataset_processing/ffhq_256_val_ids.txt`.

## Running the experiments

Every experiment has three stages: stage 1 trains the two teachers (clean and
degraded images), stage 2 learns the operator (`step2_diy`: measurement-only,
`step2_ours`: our method), and stage 3 restores the evaluation images with the
learned operator and computes the metrics. Each command checks that the
checkpoints it needs exist.

```bash
# uniform blur; kernels: mb mb2 mbcomplex gb_1 gb_3 disk aniso rgbaniso chromab
bash scripts/uniform_clean_teacher.sh
K=gb_1
for s in step1 step2_diy step2_ours step3; do bash scripts/uniform_$K.sh $s; done

# space-varying PSF field (oracle: DPIR with the true PSF field)
for s in step1 step2_diy step2_ours step3 oracle; do bash scripts/psf_field.sh $s; done

# blind super-resolution, images 1 to 100 (oracle: DPIR with the true kernel)
for s in step1 step2_diy step2_ours step3 oracle; do bash scripts/super_resolution_all.sh $s 1 100; done

# real capture
for s in step1 step2_diy step2_ours step3; do bash scripts/real_capture.sh $s; done
```

The stage-2 runs are independent across kernels and images, so they can be run
in parallel on several GPUs (`CUDA_VISIBLE_DEVICES`). In super-resolution,
stage 3 first exports the stage-2 kernel of both methods, zeroes its one-pixel
border and projects it onto the anisotropic Gaussians
(`scripts/superresolution/`), the post-processing applied to every method.

Interrupted runs resume from the last `training-state-*.pt` when the same
command is run again.

## Results

Restoration metrics are in `results/step3_*/metrics.jsonl` and
`results/oracle_*/metrics.jsonl` (keys `overall_val_loss/psnr`,
`overall_val_loss/ssim`, `overall_val_loss/lpips`, and
`overall_val_loss/psnr_matlab` for the luminance PSNR of super-resolution).
The operator error of the deblurring experiments is the shift-aligned kernel
MAE logged by stage 2 (`val_loss/kernel-mae-aligned` in
`results/step2_*/metrics.jsonl`, last value). For super-resolution:

```bash
python scripts/sr_kernel_mae_np.py \
    --pattern 'results/step2_sr_img{n}_diy/kernel_collapsed_kcrop1_gproj1.mat' \
              'results/step2_sr_img{n}_ours/kernel_collapsed_kcrop1_gproj1.mat' \
    --label diy ours --gt 'data/div2krk/gt_k_x2/kernel_{n}.mat'
```

Expected values (last checkpoint; single seed, so the last digit may move):

| | Measurement-only | Ours | True operator |
|---|---|---|---|
| Uniform, mean over 9 kernels: operator error (x1e-4) | 2.16 | **1.56** | - |
| Uniform, mean over 9 kernels: PSNR | 31.57 | **32.02** | 32.18 |
| Space-varying: operator error (x1e-4) | 8.68 | **5.52** | - |
| Space-varying: PSNR | 23.85 | **32.41** | 35.52 |
| Blind SR, 100 images: operator error (x1e-4) | 2.96 | **2.15** | - |
| Blind SR, 100 images: PSNR-Y | 33.95 | **34.99** | 35.62 |
| Blind SR, 100 images: PSNR | 32.20 | **33.21** | 33.81 |

## Compute

Stage 2 uses 4Mi images (512Ki for super-resolution). The reconstruction branch
makes operator learning 5.7 to 6.6 times slower than the measurement-only
baseline, since each operator update runs the 8-iteration DPIR solver; the
learned operator is then used exactly as the baseline's.

## Baselines

The external baselines are run from their authors' code. For the space-varying
experiment, `scripts/dump_nu_degraded.py` writes the exact degraded/clean test
pairs used by the evaluation and `scripts/score_nu_baseline.py` scores any
restoration of them with the same metric code:

```bash
python scripts/dump_nu_degraded.py --out data/nu_baseline_inputs --float
python scripts/score_nu_baseline.py --restored <dir> --clean data/nu_baseline_inputs/clean
```

## License

GNU Affero General Public License v3.0 (`LICENSE.txt`). `ddm4ip/utils/` and
`ddm4ip/networks/unets.py` include code adapted from NVIDIA's EDM2
(CC BY-NC-SA 4.0), and `ddm4ip/degradations/motion_blur.py` a motion-blur
kernel generator; their original notices are kept in the files.
