#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run step1_ffhq256_clean_patch \
    'exp=paper/step1/step1_ffhq' \
    'loss.clean_images=true' \
    'training.max_steps=${parse_nimg:"8Mi"}' \
    'training.save_every_steps=${parse_nimg:"1Mi"}' \
    'training.report_every_steps=${parse_nimg:"16Ki"}' \
    'training.plot_every_steps=${parse_nimg:"128Ki"}' \
    'training.tensorboard=true' \
    'training.seed=41'
