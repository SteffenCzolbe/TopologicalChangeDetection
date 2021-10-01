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

# brain2d dataset
#$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --semantic_loss weights/brain2d/semantic_loss/weight_decay0.0005 --recon_weight_init -11.5 --gpus -1 --accelerator dp --max_steps 16000 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --weight_decay 0.0005 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/brain2d/topology_detection/semantic_loss/weight_decay0.0005
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --recon_weight_init -5 --gpus -1 --accelerator dp --max_steps 16000 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --weight_decay 0.0005 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/brain2d/topology_detection/mse/weight_decay0.0005

# platelet-em dataset
#$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --channels 64 128 256 --semantic_loss weights/platelet-em/semantic_loss/weight_decay0.3 --recon_weight_init -11.5 --gpus -1 --accelerator dp --max_epochs 500 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --weight_decay 0.01 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/platelet-em/topology_detection/semantic_loss/weight_decay0.01
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --channels 64 128 256 --recon_weight_init -5 --gpus -1 --accelerator dp --max_epochs 500 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --weight_decay 0.01 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/platelet-em/topology_detection/mse/weight_decay0.01

