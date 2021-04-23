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

$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --prior_log_alpha 0 --trainable_alpha --prior_log_beta 0 --trainable_beta --recon_log_var 0 --trainable_recon_var --gpus -1 --accelerator dp --max_epochs 800 --lr_decline_patience 50 --early_stop_patience 300 --batch_size 32 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --prior_log_alpha -4 --trainable_alpha --prior_log_beta 8 --trainable_beta --recon_log_var -5 --trainable_recon_var --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --use_analytical_solution_for_alpha_beta --recon_log_var 0 --trainable_recon_var --gpus -1 --accelerator dp --max_epochs 600 --lr_decline_patience 100 --early_stop_patience 200 --batch_size 32 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --prior_log_alpha 2 --prior_log_beta 2 --recon_log_var -5 --trainable_recon_var --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --prior_log_alpha 0 --prior_log_beta 0 --recon_log_var -5 --trainable_recon_var --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest
$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --prior_log_alpha -2 --prior_log_beta -2 --recon_log_var -5 --trainable_recon_var --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32--bnorm --dropout --notest


