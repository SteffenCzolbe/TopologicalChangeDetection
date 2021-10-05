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

#
# Brain2d Dataset
#

# train high-capacity segmentation model for baseline "segmentation_model"
$WRAPPER_FUNC python3 -m src.baselines.train_segmentation --dataset brats2d --channels 32 64 128 256 --gpus -1 --accelerator dp --max_epochs 800 --lr_decline_patience 500 --early_stop_patience 800 --batch_size 32 --bnorm --weight_decay 0.0005 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/brain2d/baselines/segmentation

# train the deterministic registration model for baselines "jac_det_model", "li_wyatt_model"
$WRAPPER_FUNC python3 -m src.baselines.train_deterministic_registration --dataset brain2d --channels 64 128 256 --regularizer_strengh 0.1 --gpus -1 --accelerator dp --max_epochs 250 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --weight_decay 0.0005 --notest --fast_dev_run $FAST_DEV_RUN
#cp -r lightning_logs/version_0 weights/brain2d/baselines/jac_det_model
#mv lightning_logs/version_0 weights/brain2d/baselines/li_wyatt_model

# train vae model for baseline "vae_anomaly_detection"
$WRAPPER_FUNC python3 -m src.baselines.train_vae --dataset brain2d --channels 32 64 128 256 --latent_dim 512 --sigma 1 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/brain2d/baselines/vae_anomaly_detection_model


#
# Platelet-EM Dataset
#

# train high-capacity segmentation model for baseline "segmentation_model" TODO, using hand-annotation
# $WRAPPER_FUNC python3 -m src.baselines.train_segmentation --dataset platelet-em --channels 32 64 128 256 --gpus -1 --accelerator dp --max_epochs 800 --lr_decline_patience 500 --early_stop_patience 800 --batch_size 32 --bnorm --weight_decay 0.01 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/platelet-em/baselines/segmentation

# train the deterministic registration model for baselines "jac_det_model", "li_wyatt_model"
$WRAPPER_FUNC python3 -m src.baselines.train_deterministic_registration --dataset platelet-em --channels 64 128 256 --regularizer_strengh 0.1 --gpus -1 --accelerator dp --max_epochs 250 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --weight_decay 0.01 --notest --fast_dev_run $FAST_DEV_RUN
#cp -r lightning_logs/version_0 weights/platelet-em/baselines/jac_det_model
#mv lightning_logs/version_0 weights/platelet-em/baselines/li_wyatt_model

# train vae model for baseline "vae_anomaly_detection"
$WRAPPER_FUNC python3 -m src.baselines.train_vae --dataset platelet-em --channels 32 64 128 256 --latent_dim 512 --sigma 1 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --notest --fast_dev_run $FAST_DEV_RUN
#mv lightning_logs/version_0 weights/platelet-em/baselines/vae_anomaly_detection_model