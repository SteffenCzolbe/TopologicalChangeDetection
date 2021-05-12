from src.registration_model import RegistrationModel
MODELS = {
    "mse": {"path":"trained_models/mse",
            "model_cls":RegistrationModel,
            "display_name": "MSE",
            "color": "r",
            "probability_range":(2,20)},
    "semantic_loss": {"path":"trained_models/semantic_loss",
            "model_cls":RegistrationModel,
            "display_name": "Semantic Loss",
            "color": "r",
            "probability_range":(-1300,-300)},
}

FULL_MODELS = ["mse"] # only full models of our method
ALL_MODELS = ["mse"] # all models, including baselines