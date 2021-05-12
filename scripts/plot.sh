
# samples from the model
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./trained_models/mse --file plots/mse_samples
python3 -m src.eval.samples --ds1 brain2d --ds2 brats2d --weights ./trained_models/semantic_loss --file plots/semantic_loss_samples

# overlay plot
python3 -m src.eval.overlay --weights trained_models/mse --file plots/mse_overlay --range 3 20
python3 -m src.eval.overlay --weights trained_models/semantic_loss --file plots/semantic_loss_overlay --range -1300 -300

# qualitative_eval img, comparison of models
python3 -m src.eval.qualitative_comparison --file plots/qualitative_comparison
pdfcrop plots/qualitative_comparison.pdf plots/qualitative_comparison.pdf

# tumor detection
python3 -m src.eval.tumor_detection --weights  trained_models/mse --samples 32
python3 -m src.eval.tumor_detection --weights  trained_models/semantic_loss --samples 32

# ROC curves and AUC
python3 -m src.eval.roc
pdfcrop plots/roc.pdf plots/roc.pdf 