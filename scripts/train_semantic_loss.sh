#!/bin/bash

# trains the models.
FAST_DEV_RUN=0

# Check if slurm compute cluster available. Submit as slurm job if possible.
if sbatch -h &> /dev/null; then
    echo "Submitting to slurm..."
    WRAPPER_FUNC=scripts/slurm/slurm_submit.sh
else
    echo "Running locally..."
    WRAPPER_FUNC=
fi

$WRAPPER_FUNC python -m src.train_semantic_loss --dataset brain2d --channels 64 128 256 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run=$FAST_DEV_RUN

