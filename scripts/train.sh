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
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --recon_weight_init -5 --gpus -1 --accelerator dp --max_epochs 250 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN
mv lightning_logs/version_0 trained_models/mse/
cp -r trained_models/mse/ trained_auxiliary_models/vae_anomaly_detection_mse # we reuse the same model as a baseline

# train semantic loss model
$WRAPPER_FUNC python3 -m src.train_semantic_loss --dataset brain2d --channels 64 64 64 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run=$FAST_DEV_RUN
mv lightning_logs/version_0 trained_auxiliary_models/semantic_loss_feature_extractor/

# semantic loss, trainable recon, analytical prior
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --semantic_loss trained_auxiliary_models/semantic_loss_feature_extractor/ --recon_weight_init -11.5 --gpus -1 --accelerator dp --max_epochs 250 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN
mv lightning_logs/version_0 trained_models/semantic_loss/
cp -r trained_models/semantic_loss/ trained_auxiliary_models/vae_anomaly_detection_semantic # we reuse the same model as a baseline

# semantic loss, full covariance matrix in reconstruction term
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain2d --channels 64 128 256 --full_covar --semantic_loss trained_auxiliary_models/semantic_loss_feature_extractor/ --recon_weight_init -11.5 --gpus -1 --accelerator dp --max_epochs 500 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN

# train the deterministic model for baselines
$WRAPPER_FUNC python3 -m src.baselines.train_deterministic_registration --dataset brain2d --channels 64 128 256 --regularizer_strengh 0.1 --gpus -1 --accelerator dp --max_epochs 250 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --bnorm --dropout --notest
cp -r lightning_logs/version_0 trained_auxiliary_models/jac_det_model
mv lightning_logs/version_0 trained_auxiliary_models/li_wyatt_model

# train tumor segmentation model for the baselines
$WRAPPER_FUNC python3 -m src.baselines.train_segmentation --dataset brats2d --channels 32 64 128 256 --gpus -1 --accelerator dp --max_epochs 800 --lr_decline_patience 500 --early_stop_patience 800 --batch_size 32 --bnorm --dropout --notest --fast_dev_run $FAST_DEV_RUN
mv lightning_logs/version_0 trained_auxiliary_models/segmentation_model/

# train vae model for the baselines
$WRAPPER_FUNC python3 -m src.baselines.train_vae --dataset brain2d --channels 32 64 128 256 --latent_dim 512 --sigma 1 --gpus -1 --accelerator dp --max_epochs 200 --lr_decline_patience 50 --early_stop_patience 80 --batch_size 32 --notest --fast_dev_run $FAST_DEV_RUN
mv lightning_logs/version_0 trained_auxiliary_models/vae_anomaly_detection_model/