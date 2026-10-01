#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

case "${1:-}" in
step1)
    run step1_real2_seq2_clean \
        'exp=paper/step1/step1_parkinglot_x' \
        'dataset.train.clean_path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'dataset.train.noisy_path=${paths.data}/real_dataset2/real2_seq2_16.zip' \
        'dataset.test.path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'training.max_steps=${parse_nimg:"8Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.tensorboard=true' \
        'training.seed=41'
    run step1_real2_seq2_noisy \
        'exp=paper/step1/step1_parkinglot_y' \
        'dataset.train.clean_path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'dataset.train.noisy_path=${paths.data}/real_dataset2/real2_seq2_16.zip' \
        'dataset.test.path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'training.max_steps=${parse_nimg:"8Mi"}' \
        'training.save_every_steps=${parse_nimg:"1Mi"}' \
        'training.tensorboard=true' \
        'training.seed=41'
    ;;
step2_diy)
    need "$RESULTS_DIR/step1_real2_seq2_noisy/checkpoints/training-state-8388608.pt"
    run step2_real_diy \
        'exp=paper/y_cur/step2_parkinglot_y' \
        "models.pretrained_flow.path=$RESULTS_DIR/step1_real2_seq2_noisy/checkpoints/training-state-8388608.pt" \
        'dataset.train.clean_path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'dataset.train.noisy_path=${paths.data}/real_dataset2/real2_seq2_16.zip' \
        'dataset.test.path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'models.kernel.num_psfs=[9,12]' \
        'models.kernel.noise_type=gaussian' \
        'models.kernel.output_noise_sigma=0.02' \
        'models.kernel.padding=replicate' \
        '+models.kernel.kernel_norm=softplus_sum_to_one' \
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
    need "$RESULTS_DIR/step1_real2_seq2_clean/checkpoints/training-state-8388608.pt" "$RESULTS_DIR/step1_real2_seq2_noisy/checkpoints/training-state-8388608.pt"
    run step2_real_ours \
        'exp=diy_dix_pnp/step2_parkinglot_v1' \
        "models.pretrained_flow_x.path=$RESULTS_DIR/step1_real2_seq2_clean/checkpoints/training-state-8388608.pt" \
        "models.pretrained_flow_y.path=$RESULTS_DIR/step1_real2_seq2_noisy/checkpoints/training-state-8388608.pt" \
        'dataset.train.clean_path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'dataset.train.noisy_path=${paths.data}/real_dataset2/real2_seq2_16.zip' \
        'dataset.test.path=${paths.data}/real_dataset2/real2_seq2_5_6_center768.zip' \
        'models.kernel.num_psfs=[9,12]' \
        'models.kernel.noise_type=gaussian' \
        'models.kernel.output_noise_sigma=0.02' \
        'models.kernel.padding=replicate' \
        'models.kernel.kernel_norm=softplus_sum_to_one' \
        'loss.restorer.sigma=0.02' \
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
    need "$RESULTS_DIR/step2_real_diy/checkpoints/training-state-4194304.pt"
    run step3_real_diy \
        'exp=paper/step3/y_cur/step3_parkinglot_diy' \
        "models.kernel.path=$RESULTS_DIR/step2_real_diy/checkpoints/training-state-4194304.pt" \
        'models.kernel.type=direct_center' \
        'models.kernel.num_psfs=[9,12]' \
        'models.kernel.kernel_size=13' \
        'models.kernel.kernel_ch=3' \
        'models.kernel.noise_type=gaussian' \
        'models.kernel.output_noise_sigma=0.02' \
        'models.deepinv_solver.method=dpir' \
        'dataset.test.path=${paths.data}/real_dataset2/real2_seq2_16.zip' \
        'training.max_steps=1' \
        '+training.save_corrupt=true' \
        'models.kernel.padding=replicate' \
        '+models.kernel.kernel_norm=softplus_sum_to_one' \
        '+models.kernel.auto_center=true' \
        '+models.kernel.auto_center_window=10' \
        '+models.kernel.auto_center_anchor_channel=1' \
        'loss.patch_batch_size=64'
    need "$RESULTS_DIR/step2_real_ours/checkpoints/training-state-4194304.pt"
    run step3_real_ours \
        'exp=paper/step3/y_cur/step3_parkinglot_diy' \
        "models.kernel.path=$RESULTS_DIR/step2_real_ours/checkpoints/training-state-4194304.pt" \
        'models.kernel.type=direct_center' \
        'models.kernel.num_psfs=[9,12]' \
        'models.kernel.kernel_size=13' \
        'models.kernel.kernel_ch=3' \
        'models.kernel.noise_type=gaussian' \
        'models.kernel.output_noise_sigma=0.02' \
        'models.deepinv_solver.method=dpir' \
        'dataset.test.path=${paths.data}/real_dataset2/real2_seq2_16.zip' \
        'training.max_steps=1' \
        '+training.save_corrupt=true' \
        'models.kernel.padding=replicate' \
        '+models.kernel.kernel_norm=softplus_sum_to_one' \
        '+models.kernel.auto_center=true' \
        '+models.kernel.auto_center_window=10' \
        '+models.kernel.auto_center_anchor_channel=1' \
        'loss.patch_batch_size=64'
    ;;
*)
    echo "usage: $0 step1|step2_diy|step2_ours|step3" >&2; exit 1 ;;
esac
