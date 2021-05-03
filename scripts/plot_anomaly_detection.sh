python3 -m src.qualitative_eval --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/fixed_prior --file plots/fixed_prior_brain2brats
python3 -m src.qualitative_eval --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/analytical_prior --file plots/analytical_prior_brain2brats
#python3 -m src.qualitative_eval --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/analytical_prior_semantic_augmentation --file plots/analytical_semantic_augmentation_brain2brats
python3 -m src.qualitative_eval --ds1 brain2d --ds2 brats2d --weights ./lightning_logs/analytical_prior_semantic_loss --file plots/analytical_semantic_loss_brain2brats

python3 -m src.qualitative_eval --ds1 brats2d --ds2 brain2d --weights ./lightning_logs/fixed_prior --file plots/fixed_prior_brats2brain
python3 -m src.qualitative_eval --ds1 brats2d --ds2 brain2d --weights ./lightning_logs/analytical_prior --file plots/analytical_prior_brats2brain
#python3 -m src.qualitative_eval --ds1 brats2d --ds2 brain2d --weights ./lightning_logs/analytical_prior_semantic_augmentation --file plots/analytical_semantic_augmentation_brats2brain
python3 -m src.qualitative_eval --ds1 brats2d --ds2 brain2d --weights ./lightning_logs/analytical_prior_semantic_loss --file plots/analytical_semantic_loss_brats2brain