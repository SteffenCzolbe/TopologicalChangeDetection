python3 -m src.qualitative_eval --weights ./lightning_logs/fixed_prior --file plots/fixed_prior
python3 -m src.qualitative_eval --weights ./lightning_logs/trainable_prior --file plots/trainable_prior
python3 -m src.qualitative_eval --weights ./lightning_logs/fixed_model_var --file plots/fixed_model_var
python3 -m src.qualitative_eval --weights ./lightning_logs/analytical_prior --file plots/analytical_prior