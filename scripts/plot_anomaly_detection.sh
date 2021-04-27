python3 -m src.qualitative_eval --weights ./lightning_logs/fixed_alpha_beta_0 --file plots/fixed_alpha_beta_0
python3 -m src.qualitative_eval --weights ./lightning_logs/trainable_alpha_beta --file plots/trainable_alpha_beta
python3 -m src.qualitative_eval --weights ./lightning_logs/fixed_model_var --file plots/fixed_model_var
python3 -m src.qualitative_eval --weights ./lightning_logs/analytical_alpha_beta_ppix_expect --file plots/analytical_alpha_beta_ppix_expect