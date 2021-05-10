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

# trainable recon, analytical prior
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --analytical_prior --trainable_recon --recon_weight_init -5 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN

# semantic loss, trainable recon, analytical prior
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --analytical_prior --trainable_recon --semantic_loss trained_models/semantic_loss --recon_weight_init -12.5 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN

# fixed prior parameters alpha, beta
$WRAPPER_FUNC python -m src.train_registration --dataset brain2d --channels 64 128 256 --prior_weights_init 0 0 --recon_weight_init -5 --trainable_recon --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN
