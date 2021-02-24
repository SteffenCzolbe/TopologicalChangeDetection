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

$WRAPPER_FUNC python -m src.train_registration --dataset mnist --lam 0.5 --channels 64 128 --batch_size 32 --gpus 1 --max_epochs 80 --lr_decline_patience 10 --early_stop_patience 25 --conv_layers_per_stage 1 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset mnist --lam 0.75 --channels 64 128 --batch_size 32 --gpus 1 --max_epochs 80 --lr_decline_patience 10 --early_stop_patience 25 --conv_layers_per_stage 1 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset mnist --lam 1 --channels 64 128 --batch_size 32 --gpus 1 --max_epochs 80 --lr_decline_patience 10 --early_stop_patience 25 --conv_layers_per_stage 1 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset mnist --lam 1.5 --channels 64 128 --batch_size 32 --gpus 1 --max_epochs 80 --lr_decline_patience 10 --early_stop_patience 25 --conv_layers_per_stage 1 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset mnist --lam 2 --channels 64 128 --batch_size 32 --gpus 1 --max_epochs 80 --lr_decline_patience 10 --early_stop_patience 25 --conv_layers_per_stage 1 --bnorm --dropout --notest
