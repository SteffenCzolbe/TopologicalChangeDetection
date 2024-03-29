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

# platelet with weight decay 0.3
$WRAPPER_FUNC python3 -m src.train_semantic_loss --dataset platelet-em --channels 32 64 64 --gpus -1 --accelerator dp --max_steps 2000 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --weight_decay 0.03 --notest
mv lightning_logs/version_0 weights/platelet-em/semantic_loss/weight_decay0.3

# brain with weight decay of 0.0005
$WRAPPER_FUNC python3 -m src.train_semantic_loss --dataset brain2d --channels 32 64 64 --gpus -1 --accelerator dp --max_steps 20000 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 16 --bnorm --weight_decay 0.0005 --notest
mv lightning_logs/version_0 weights/brain2d/semantic_loss/weight_decay0.0005
