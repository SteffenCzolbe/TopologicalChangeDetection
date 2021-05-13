from src.registration_model import RegistrationModel
MODELS = {
    "mse": {"path": "trained_models/mse",
            "model_cls": RegistrationModel,
            "display_name": "MSE",
            "color": "r",
            "probability_range": (1.5, 10)},
    "semantic_loss": {"path": "trained_models/semantic_loss",
                      "model_cls": RegistrationModel,
                      "display_name": "Semantic Loss",
                      "color": "r",
                      "probability_range": (-1900, -500)},
}

FULL_MODELS = ["mse", "semantic_loss"]  # only full models of our method
ALL_MODELS = ["mse", "semantic_loss"]  # all models, including baselines
