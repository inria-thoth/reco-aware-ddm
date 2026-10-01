#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

case "${1:-}" in
step1)
    run step1_ffhq256_chromab_patch \
        'exp=paper/step1/step1_ffhq256_chromab_y' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step2_diy)
    need "$RESULTS_DIR/step1_ffhq256_chromab_patch/checkpoints/training-state-8388608.pt"
    run step2_ffhq256_chromab_diy \
        'exp=paper/y_cur/step2_ffhq256_chromab_diy' \
        'models.kernel.padding=replicate' \
        'models.kernel.kernel_norm=softplus_sum_to_one' \
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
        'models.kernel.auto_center=true' \
        'models.kernel.auto_center_window=10' \
        '+loss.ema_kernel_beta=0.999' \
        'training.tensorboard=true' \
        'training.seed=41' \
        '+loss.compile=true' \
        '+training.tf32=true'
    ;;
step2_ours)
    need "$RESULTS_DIR/step1_ffhq256_clean_patch/checkpoints/training-state-8388608.pt" "$RESULTS_DIR/step1_ffhq256_chromab_patch/checkpoints/training-state-8388608.pt"
    run step2_ffhq256_chromab_ours \
        'exp=diy_dix_pnp/step2_ffhq256_chromab_v1' \
        "models.pretrained_flow_x.path=$RESULTS_DIR/step1_ffhq256_clean_patch/checkpoints/training-state-8388608.pt" \
        "models.pretrained_flow_y.path=$RESULTS_DIR/step1_ffhq256_chromab_patch/checkpoints/training-state-8388608.pt" \
        'models.kernel.padding=replicate' \
        'models.kernel.kernel_norm=softplus_sum_to_one' \
        'loss.restorer.grad_iters=1' \
        '+loss.restorer.max_iter=8' \
        'training.batch_size=32' \
        'optim.kernel.lr=0.003' \
        '+loss.ema_kernel_beta=0.999' \
        '+optim.kernel.lr_scheduler.type=CosineAnnealingLR' \
        '+optim.kernel.lr_scheduler.batch_size=32' \
        '+optim.kernel.lr_scheduler.rampup_nimg=${parse_nimg:"200Ki"}' \
        '+optim.kernel.lr_scheduler.max_nimg=${parse_nimg:"4Mi"}' \
        '+optim.kernel.lr_scheduler.eta_min=1e-7' \
        'training.max_steps=${parse_nimg:"4Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.report_every_steps=${parse_nimg:"16Ki"}' \
        'training.plot_every_steps=${parse_nimg:"32Ki"}' \
        'models.kernel.auto_center=true' \
        'models.kernel.auto_center_window=10' \
        '+loss.compile=true' \
        '+loss.restorer.compile=true' \
        '+training.tf32=true' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step3)
    need "$RESULTS_DIR/step2_ffhq256_chromab_diy/checkpoints/training-state-4194304.pt"
    run step3_ffhq256_chromab_diy \
        'exp=paper/step3/y_cur/step3_ffhq256_chromab_diy' \
        "models.kernel.path=$RESULTS_DIR/step2_ffhq256_chromab_diy/checkpoints/training-state-4194304.pt" \
        'models.deepinv_solver.method=dpir' \
        '+models.kernel.use_ema=true' \
        '+models.kernel.auto_center=true' \
        '+models.kernel.auto_center_anchor_channel=1' \
        '+models.kernel.auto_center_window=10'
    need "$RESULTS_DIR/step2_ffhq256_chromab_ours/checkpoints/training-state-4194304.pt"
    run step3_ffhq256_chromab_ours \
        'exp=paper/step3/y_cur/step3_ffhq256_chromab_diy' \
        "models.kernel.path=$RESULTS_DIR/step2_ffhq256_chromab_ours/checkpoints/training-state-4194304.pt" \
        'models.deepinv_solver.method=dpir' \
        '+models.kernel.use_ema=true' \
        '+models.kernel.auto_center=true' \
        '+models.kernel.auto_center_anchor_channel=1' \
        '+models.kernel.auto_center_window=10'
    ;;
*)
    echo "usage: $0 step1|step2_diy|step2_ours|step3" >&2; exit 1 ;;
esac
