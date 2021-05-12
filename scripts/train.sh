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
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --recon_weight_init -5 --gpus -1 --accelerator dp --max_epochs 150 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN
mv lightning_logs/version_0 trained_models/mse/

# train semantic loss model
$WRAPPER_FUNC python3 -m src.train_semantic_loss --dataset brain2d --channels 64 64 64 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run=$FAST_DEV_RUN
mv lightning_logs/version_0 trained_auxiliary_models/semantic_loss_feature_extractor/

# semantic loss, trainable recon, analytical prior
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --semantic_loss trained_auxiliary_models/semantic_loss_feature_extractor/ --recon_weight_init -11.5 --gpus -1 --accelerator dp --max_epochs 150 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN
mv lightning_logs/version_0 trained_models/semantic_loss/