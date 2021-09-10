# introduction figure template
python3 -m src.eval.introfig
pdfcrop plots/intro.pdf plots/intro.pdf

# samples from the model
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./trained_models/mse --file plots/mse_samples
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./trained_models/semantic_loss --file plots/semantic_loss_samples
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./trained_models/semantic_loss_full_covar --file plots/semantic_loss_full_covar_samples

# overlay plot
python3 -m src.eval.overlay --weights trained_models/mse --file plots/mse_overlay --range 3 20
python3 -m src.eval.overlay --weights trained_models/semantic_loss --file plots/semantic_loss_overlay --range -1300 -300

# mean Lsym
python3 -m src.eval.average_topological_difference --file plots/mean_Lsym_semantic_loss --model_name semantic_loss --samples 64
pdfcrop plots/mean_Lsym_semantic_loss.pdf plots/mean_Lsym_semantic_loss.pdf
pdfcrop plots/mean_Lsym_semantic_loss_boxplot.pdf plots/mean_Lsym_semantic_loss_boxplot.pdf

# tumor detection
python3 -m src.eval.tumor_detection --model_name  mse --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  semantic_loss --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  semantic_loss_full_covar --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  vae_anomaly_detection_mse --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  vae_anomaly_detection_semantic_loss --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  jac_det_model --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  li_wyatt_model --samples 32 --non_cached
python3 -m src.eval.tumor_detection_for_segmentation_model --model_name  segmentation_model --non_cached
python3 -m src.eval.tumor_detection_for_anomaly_detection_model --model_name  vae_anomaly_detection --non_cached

# plot publication fig
python3 -m src.eval.publication_fig --file plots/pub_fig
pdfcrop plots/pub_fig.pdf plots/pub_fig.pdf

# ROC curves and AUC
python3 -m src.eval.roc --file plots/roc_tumor --bootstrap_sample_cnt 8
pdfcrop plots/roc_tumor.pdf plots/roc_tumor.pdf 
python3 -m src.eval.roc --file plots/roc_tumor_edema --include_edema --bootstrap_sample_cnt 8
pdfcrop plots/roc_tumor_edema.pdf plots/roc_tumor_edema.pdf 