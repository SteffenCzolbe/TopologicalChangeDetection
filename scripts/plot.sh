
# samples from the model
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/analytical_prior --file plots/analytical_prior_samples
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/semantic_loss --file plots/semantic_loss_samples

# overlay plot
python3 -m src.eval.overlay --weights lightning_logs/analytical_prior --file plots/analytical_prior_overlay --range 1 20
python3 -m src.eval.overlay --weights lightning_logs/semantic_loss --file plots/semantic_loss_overlay --range -50 20

# violin plot
python3 -m src.eval.violin_plot --weights lightning_logs/analytical_prior --file plots/analytical_prior_violin
python3 -m src.eval.violin_plot --weights lightning_logs/semantic_loss --file plots/semantic_loss_violin

# violin plot segmentation missmatch
python3 -m src.eval.violin_seg_match --weights lightning_logs/analytical_prior --file plots/analytical_prior_violin_seg_match
python3 -m src.eval.violin_seg_match --weights lightning_logs/semantic_loss --file plots/semantic_loss_violin_seg_match

# ROC and AUC
python3 -m src.eval.roc