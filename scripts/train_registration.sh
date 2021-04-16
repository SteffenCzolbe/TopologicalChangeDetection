#!/bin/bash

# trains the models.

# Check if slurm compute cluster available. Submit as slurm job if possible.
if sbatch -h &> /dev/null; then
    echo "Submitting to slurm..."
    WRAPPER_FUNC=scripts/slurm/slurm_submit.sh
else
    echo "Running locally..."
    WRAPPER_FUNC=
fi

$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --batch_size 32 --gpus -1 --accelerator dp --max_epochs 800 --lr_decline_patience 20 --early_stop_patience 150 --conv_layers_per_stage 1 --bnorm --dropout --notest
#$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --batch_size 32 --gpus -1 --accelerator dp --max_epochs 80 --lr_decline_patience 10 --early_stop_patience 25 --conv_layers_per_stage 1 --bnorm --dropout --integrate --notest
#$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --batch_size 32 --gpus -1 --accelerator dp --max_epochs 80 --lr_decline_patience 10 --early_stop_patience 25 --conv_layers_per_stage 1 --bnorm --dropout --per_pixel_prior --notest