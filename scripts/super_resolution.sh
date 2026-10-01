#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

N="${2:-}"
[ -n "$N" ] || { echo "usage: $0 step1|step2_diy|step2_ours|step3|oracle IMAGE (1-100)" >&2; exit 1; }

case "${1:-}" in
step1)
    run step1_sr_x_img${N} \
        'exp=paper/step1/step1_sr_y' \
        "dataset.train.clean_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.train.noisy_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.test.path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        'dataset/degradation@dataset.train.degradation=file_downsampling' \
        'dataset/degradation@dataset.test.degradation=file_downsampling' \
        "dataset.train.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        "dataset.test.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        'dataset.train.degradation.padding=replicate' \
        'dataset.test.degradation.padding=replicate' \
        'dataset.train.degradation.kernel_size=15' \
        'dataset.test.degradation.kernel_size=15' \
        'dataset.train.noise.std=0' \
        'dataset.test.noise.std=0' \
        'dataset.train.patch_size=64' \
        'dataset.test.patch_size=64' \
        'loss.clean_images=true' \
        'ema.stds=[0.004]' \
        'training.batch_size=32' \
        'loss.n_accum_steps=4' \
        'optim.lr=0.01' \
        'training.max_steps=${parse_nimg:"1Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.report_every_steps=${parse_nimg:"32Ki"}' \
        'training.plot_every_steps=${parse_nimg:"128Ki"}' \
        'training.tensorboard=true' \
        'training.seed=41'
    run step1_sr_y_img${N}_ss \
        'exp=paper/step1/step1_sr_y' \
        'dataset.train.degradation.kind=none' \
        '+dataset.train.noisy_patch_size=32' \
        "dataset.train.clean_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.train.noisy_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.test.path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        'dataset/degradation@dataset.test.degradation=file_downsampling' \
        "dataset.test.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        'dataset.test.degradation.padding=replicate' \
        'dataset.test.degradation.kernel_size=15' \
        'dataset.train.noise.std=0' \
        'dataset.test.noise.std=0' \
        'dataset.train.patch_size=64' \
        'dataset.test.patch_size=64' \
        'ema.stds=[0.004]' \
        'training.batch_size=32' \
        'loss.n_accum_steps=4' \
        'optim.lr=0.01' \
        'training.max_steps=${parse_nimg:"1Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.report_every_steps=${parse_nimg:"32Ki"}' \
        'training.plot_every_steps=${parse_nimg:"128Ki"}' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step2_diy)
    need "$RESULTS_DIR/step1_sr_y_img${N}_ss/checkpoints/training-state-1048576.pt"
    run step2_sr_img${N}_diy \
        'exp=step2_sr_y' \
        'models@models.pretrained_flow=xxxs_flow_32' \
        'models.kernel.kernel_size=11' \
        '++loss.compile=true' \
        '++training.tf32=true' \
        'models.kernel.padding=zeros_normalized' \
        'dataset.train.patch_size=64' \
        'dataset.test.patch_size=64' \
        'dataset.train.degradation.kind=none' \
        '+dataset.train.noisy_patch_size=32' \
        "models.pretrained_flow.path=$RESULTS_DIR/step1_sr_y_img${N}_ss/checkpoints/training-state-1048576.pt" \
        "dataset.train.clean_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.train.noisy_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.test.path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        'dataset/degradation@dataset.test.degradation=file_downsampling' \
        "dataset.test.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        'dataset.test.degradation.padding=circular' \
        'dataset.test.degradation.kernel_size=11' \
        'dataset.train.noise.std=0' \
        'dataset.test.noise.std=0' \
        'training.batch_size=32' \
        'optim.aux.lr=0.00001' \
        'optim.kernel.lr=0.003' \
        'training.max_steps=${parse_nimg:"513Ki"}' \
        'training.save_every_steps=${parse_nimg:"512Ki"}' \
        'training.report_every_steps=${parse_nimg:"32Ki"}' \
        'training.plot_every_steps=${parse_nimg:"128Ki"}' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step2_ours)
    need "$RESULTS_DIR/step1_sr_x_img${N}/checkpoints/training-state-1048576.pt" "$RESULTS_DIR/step1_sr_y_img${N}_ss/checkpoints/training-state-1048576.pt"
    run step2_sr_img${N}_ours \
        'exp=diy_dix_pnp/step2_div2krk_sr' \
        'models.kernel.kernel_size=11' \
        'models.kernel.padding=zeros_normalized' \
        'dataset.train.degradation.kind=none' \
        '+dataset.train.noisy_patch_size=32' \
        "models.pretrained_flow_x.path=$RESULTS_DIR/step1_sr_x_img${N}/checkpoints/training-state-1048576.pt" \
        "models.pretrained_flow_y.path=$RESULTS_DIR/step1_sr_y_img${N}_ss/checkpoints/training-state-1048576.pt" \
        "dataset.train.clean_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.train.noisy_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.test.path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        'dataset/degradation@dataset.test.degradation=file_downsampling' \
        "dataset.test.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        'dataset.test.degradation.padding=circular' \
        'dataset.test.degradation.kernel_size=11' \
        'dataset.train.noise.std=0' \
        'dataset.test.noise.std=0' \
        '++loss.compile=true' \
        '+loss.restorer.compile=true' \
        '++training.tf32=true' \
        '+optim.kernel.lr_scheduler.type=CosineAnnealingLR' \
        '+optim.kernel.lr_scheduler.batch_size=32' \
        '+optim.kernel.lr_scheduler.rampup_nimg=${parse_nimg:"25Ki"}' \
        '+optim.kernel.lr_scheduler.max_nimg=${parse_nimg:"512Ki"}' \
        '+optim.kernel.lr_scheduler.eta_min=1e-7' \
        'loss.lambda_dix=1.0' \
        '+loss.ema_kernel_beta=0.999' \
        'training.batch_size=32' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step3)
    for arm in diy ours; do
        d="$RESULTS_DIR/step2_sr_img${N}_${arm}"
        snap="$(ls -1 "$d"/checkpoints/network-snapshot-*.pkl | sort -V | tail -1)"
        python scripts/superresolution/get_kernel_from_network.py --network="$snap" --output-file="$d/kernel_collapsed.mat"
        python scripts/superresolution/crop_kernel_border.py --in "$d/kernel_collapsed.mat" --out "$d/kernel_collapsed_kcrop1.mat" --border 1
        python scripts/superresolution/project_kernel_gaussian.py --in "$d/kernel_collapsed_kcrop1.mat" --out "$d/kernel_collapsed_kcrop1_gproj1.mat" --alpha 1
    done
    need "$RESULTS_DIR/step2_sr_img${N}_diy/kernel_collapsed_kcrop1_gproj1.mat"
    run step3_sr_img${N}_diy \
        'exp=paper/step3/sr/step3_div2krk_sr' \
        "dataset.test.path=\${paths.data}/div2krk/gt/img_${N}_gt.png" \
        "dataset.test.noisy_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.test.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        "models.kernel.fixed_kernel_path=$RESULTS_DIR/step2_sr_img${N}_diy/kernel_collapsed_kcrop1_gproj1.mat" \
        'models.kernel.padding=zeros_normalized' \
        'training.tensorboard=true'
    need "$RESULTS_DIR/step2_sr_img${N}_ours/kernel_collapsed_kcrop1_gproj1.mat"
    run step3_sr_img${N}_ours \
        'exp=paper/step3/sr/step3_div2krk_sr' \
        "dataset.test.path=\${paths.data}/div2krk/gt/img_${N}_gt.png" \
        "dataset.test.noisy_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.test.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        "models.kernel.fixed_kernel_path=$RESULTS_DIR/step2_sr_img${N}_ours/kernel_collapsed_kcrop1_gproj1.mat" \
        'training.tensorboard=true'
    ;;
oracle)
    run oracle_sr_img${N} \
        'exp=paper/step3/sr/step3_div2krk_sr' \
        "dataset.test.path=\${paths.data}/div2krk/gt/img_${N}_gt.png" \
        "dataset.test.noisy_path=\${paths.data}/div2krk/lr_x2/im_${N}.png" \
        "dataset.test.degradation.kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        "models.kernel.fixed_kernel_path=\${paths.data}/div2krk/gt_k_x2/kernel_${N}.mat" \
        'training.tensorboard=true'
    ;;
*)
    echo "usage: $0 step1|step2_diy|step2_ours|step3|oracle IMAGE (1-100)" >&2; exit 1 ;;
esac
