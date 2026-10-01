#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

case "${1:-}" in
step1)
    run step1_div2k_nu_clean_patch \
        'exp=paper/step1/step1_nonuniform' \
        'loss.clean_images=true' \
        'dataset.train.patch_size=64' \
        'dataset.test.patch_size=64' \
        'training.max_steps=${parse_nimg:"8Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.report_every_steps=${parse_nimg:"16Ki"}' \
        'training.plot_every_steps=${parse_nimg:"128Ki"}' \
        'training.tensorboard=true' \
        'training.seed=41'
    run step1_div2k_nu_y_patch \
        'exp=paper/step1/step1_nonuniform_y' \
        'loss.clean_images=false' \
        'dataset.train.patch_size=64' \
        'dataset.test.patch_size=64' \
        'training.max_steps=${parse_nimg:"8Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.report_every_steps=${parse_nimg:"16Ki"}' \
        'training.plot_every_steps=${parse_nimg:"128Ki"}' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step2_diy)
    need "$RESULTS_DIR/step1_div2k_nu_y_patch/checkpoints/training-state-8388608.pt"
    run step2_nu_diy \
        'exp=paper/y_cur/step2_div2k_nu_diy' \
        "models.pretrained_flow.path=$RESULTS_DIR/step1_div2k_nu_y_patch/checkpoints/training-state-8388608.pt" \
        'models.kernel.padding=replicate' \
        'models.kernel.kernel_norm=softplus_sum_to_one' \
        '+models.kernel.auto_center=true' \
        '+models.kernel.auto_center_window=10' \
        '+models.kernel.auto_center_anchor_channel=1' \
        'training.batch_size=32' \
        'optim.kernel.lr=0.003' \
        '+optim.kernel.lr_scheduler.type=CosineAnnealingLR' \
        '+optim.kernel.lr_scheduler.batch_size=32' \
        '+optim.kernel.lr_scheduler.rampup_nimg=${parse_nimg:"200Ki"}' \
        '+optim.kernel.lr_scheduler.max_nimg=${parse_nimg:"4Mi"}' \
        '+optim.kernel.lr_scheduler.eta_min=1e-7' \
        'training.max_steps=${parse_nimg:"4Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.report_every_steps=${parse_nimg:"16Ki"}' \
        'training.plot_every_steps=${parse_nimg:"32Ki"}' \
        'training.max_val_batches=16' \
        '+loss.ema_kernel_beta=0.999' \
        '+loss.compile=true' \
        '+training.tf32=true' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step2_ours)
    need "$RESULTS_DIR/step1_div2k_nu_clean_patch/checkpoints/training-state-8388608.pt" "$RESULTS_DIR/step1_div2k_nu_y_patch/checkpoints/training-state-8388608.pt"
    run step2_nu_ours \
        'exp=diy_dix_pnp/step2_div2k_nu_v1' \
        "models.pretrained_flow_x.path=$RESULTS_DIR/step1_div2k_nu_clean_patch/checkpoints/training-state-8388608.pt" \
        "models.pretrained_flow_y.path=$RESULTS_DIR/step1_div2k_nu_y_patch/checkpoints/training-state-8388608.pt" \
        'models.kernel.padding=replicate' \
        'models.kernel.kernel_norm=softplus_sum_to_one' \
        'loss.restorer.grad_iters=1' \
        '+loss.restorer.max_iter=8' \
        '+models.kernel.auto_center=true' \
        '+models.kernel.auto_center_window=10' \
        '+models.kernel.auto_center_anchor_channel=1' \
        'training.batch_size=32' \
        'optim.kernel.lr=0.003' \
        '+optim.kernel.lr_scheduler.type=CosineAnnealingLR' \
        '+optim.kernel.lr_scheduler.batch_size=32' \
        '+optim.kernel.lr_scheduler.rampup_nimg=${parse_nimg:"200Ki"}' \
        '+optim.kernel.lr_scheduler.max_nimg=${parse_nimg:"4Mi"}' \
        '+optim.kernel.lr_scheduler.eta_min=1e-7' \
        'training.max_steps=${parse_nimg:"4Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.report_every_steps=${parse_nimg:"16Ki"}' \
        'training.plot_every_steps=${parse_nimg:"32Ki"}' \
        'training.max_val_batches=16' \
        '+loss.ema_kernel_beta=0.999' \
        '+loss.compile=true' \
        '+loss.restorer.compile=true' \
        '+training.tf32=true' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step3)
    need "$RESULTS_DIR/step2_nu_diy/checkpoints/training-state-4194304.pt"
    run step3_nu_diy \
        'exp=paper/step3/y_cur/step3_ffhq256_nu_diy' \
        "models.kernel.path=$RESULTS_DIR/step2_nu_diy/checkpoints/training-state-4194304.pt" \
        'models.kernel.type=direct_center' \
        'models.kernel.num_psfs=[8,8]' \
        'models.kernel.kernel_size=27' \
        'models.kernel.kernel_ch=3' \
        'models.kernel.noise_type=gaussian' \
        'models.deepinv_solver.method=dpir' \
        '+models.kernel.use_ema=true' \
        '++models.kernel.auto_center=true' \
        '++models.kernel.auto_center_window=10' \
        '++models.kernel.auto_center_anchor_channel=1' \
        'training.save_eval_to_file=false' \
        'loss.patch_batch_size=64'
    need "$RESULTS_DIR/step2_nu_ours/checkpoints/training-state-4194304.pt"
    run step3_nu_ours \
        'exp=paper/step3/y_cur/step3_ffhq256_nu_diy' \
        "models.kernel.path=$RESULTS_DIR/step2_nu_ours/checkpoints/training-state-4194304.pt" \
        'models.kernel.type=direct_center' \
        'models.kernel.num_psfs=[8,8]' \
        'models.kernel.kernel_size=27' \
        'models.kernel.kernel_ch=3' \
        'models.kernel.noise_type=gaussian' \
        'models.deepinv_solver.method=dpir' \
        '+models.kernel.use_ema=true' \
        '++models.kernel.auto_center=true' \
        '++models.kernel.auto_center_window=10' \
        '++models.kernel.auto_center_anchor_channel=1' \
        'training.save_eval_to_file=false' \
        'loss.patch_batch_size=64'
    ;;
oracle)
    run oracle_nu \
        'exp=paper/step3/y_cur/step3_ffhq256_nu_diy' \
        'models.kernel.path=null' \
        'models.kernel.initialization=from_psf' \
        'models.kernel.psf_path=${paths.data}/psfs/8x8_realpsf_rgb_27.pt' \
        'models.kernel.kernel_norm=none' \
        'models.kernel.auto_center=false' \
        'loss.patch_batch_size=64' \
        'training.tensorboard=true'
    ;;
*)
    echo "usage: $0 step1|step2_diy|step2_ours|step3|oracle" >&2; exit 1 ;;
esac
