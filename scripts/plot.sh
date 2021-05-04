
# samples from the model
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/analytical_prior --file plots/analytical_prior_samples
#python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/analytical_prior_semantic_loss --file plots/analytical_prior_semantic_loss

# overlay plot
python3 -m src.eval.overlay --weights lightning_logs/analytical_prior --file plots/analytical_prior_overlay


# violin plot
python3 -m src.eval.violin_plot --weights lightning_logs/analytical_prior --file plots/analytical_prior_violin