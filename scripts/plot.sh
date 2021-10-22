BRAIN2DMSEMODEL=./weights/brain2d/topology_detection/mse/weight_decay0.0005
BRAIN2DSEMMODEL=./weights/brain2d/topology_detection/semantic_loss/weight_decay0.0005
PLTELETEMMSEMODEL=./weights/platelet-em/topology_detection/mse/weight_decay0.01
PLTELETEMSEMMODEL=./weights/platelet-em/topology_detection/semantic_loss/weight_decay0.01

# samples from the model (dev purposes)
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights $BRAIN2DMSEMODEL --file plots/brain2d_mse_samples 
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights $BRAIN2DSEMMODEL --file plots/brain2d_semantic_loss_samples
python3 -m src.eval.samples --ds1 platelet-em --ds2 platelet-em --weights $PLTELETEMSEMMODEL --file plots/plateletem_semantic_loss_samples


# overlay plot (dev purposes, use to tune heatmap value range)
python3 -m src.eval.overlay --ds1 brain2d --ds2 brats2d --weights trained_models/mse --file plots/brain2d_mse_overlay --range 3 20
python3 -m src.eval.overlay --ds1 brain2d --ds2 brats2d --weights $BRAIN2DSEMMODEL --file plots/brain2d_semantic_loss_overlay --range -1300 -300
python3 -m src.eval.overlay --ds1 platelet-em --ds2 platelet-em --weights $PLTELETEMSEMMODEL --file plots/plateletem_semantic_loss_overlay --range -85 -55

# mean Lsym in brain by region (publication fig)
python3 -m src.eval.average_topological_difference --file plots/mean_Lsym_semantic_loss --model_name semantic_loss --samples 64
pdfcrop plots/mean_Lsym_semantic_loss.pdf plots/mean_Lsym_semantic_loss.pdf
pdfcrop plots/mean_Lsym_semantic_loss_boxplot.pdf plots/mean_Lsym_semantic_loss_boxplot.pdf

# run tumor detection (data pre-preprocessing for brains)
python3 -m src.eval.tumor_detection --model_name  mse --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  semantic_loss --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  jac_det_model --samples 32 --non_cached
python3 -m src.eval.tumor_detection --model_name  li_wyatt_model --samples 32 --non_cached
python3 -m src.eval.tumor_detection_for_segmentation_model --model_name  segmentation_model --non_cached
python3 -m src.eval.tumor_detection_for_anomaly_detection_model --model_name  vae_anomaly_detection --non_cached

# plot brain model samples (publication fig)
python3 -m src.eval.brain_samples_pub_fig --dataset brain2d --file plots/brain2d_samples_comparison
pdfcrop plots/brain2d_samples_comparison.pdf plots/brain2d_samples_comparison.pdf

# plot platelet model samples (publication fig)
python3 -m src.eval.platelet_samples_pub_fig --dataset platelet-em --file plots/plateletem_samples_comparison
pdfcrop plots/plateletem_samples_comparison.pdf plots/plateletem_samples_comparison.pdf

# introduction fig (platelet)
python3 -m src.eval.platelet_intro_fig --file plots/intro_fig
pdfcrop plots/intro_fig.pdf plots/intro_fig.pdf
python3 -m src.eval.platelet_intro_fig --overlay_contour --file plots/intro_fig_contour
pdfcrop plots/intro_fig_contour.pdf plots/intro_fig_contour.pdf
python3 -m src.eval.platelet_intro_fig --overlay_seg --file plots/intro_fig_with_segmentation
pdfcrop plots/intro_fig_with_segmentation.pdf plots/intro_fig_with_segmentation.pdf

# introduction fig (brains)
python3 -m src.eval.brain_intro_fig --file plots/brain_intro_fig
pdfcrop plots/brain_intro_fig.pdf plots/brain_intro_fig.pdf
python3 -m src.eval.brain_intro_fig --overlay_tumor --file plots/brain_intro_fig_contour
pdfcrop plots/brain_intro_fig_contour.pdf plots/brain_intro_fig_contour.pdf

# ROC curves and AUC (publication fig)
python3 -m src.eval.roc --dataset brain2d --file plots/brain2d_roc_tumor --bootstrap_sample_cnt 8
pdfcrop plots/brain2d_roc_tumor.pdf plots/brain2d_roc_tumor.pdf 
python3 -m src.eval.roc --dataset platelet-em --file plots/platelet-em_roc --bootstrap_sample_cnt 8
pdfcrop plots/platelet-em_roc.pdf plots/platelet-em_roc.pdf 

# test registration models
python3 -m src.test_registration --load_from_checkpoint $PLTELETEMSEMMODEL
python3 -m src.test_registration --load_from_checkpoint $PLTELETEMMSEMODEL
python3 -m src.test_registration --load_from_checkpoint $BRAIN2DSEMMODEL
python3 -m src.test_registration --load_from_checkpoint $BRAIN2DMSEMODEL