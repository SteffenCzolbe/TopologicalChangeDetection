from src.registration_model import RegistrationModel
from src.baselines.vae_anomaly_detection import VAEAnaomalyDetection
from src.baselines.jac_det_model import JacDetModel
from src.baselines.li_wyatt_model import LiWyattModel
from src.baselines.segmentation_model import SegmentationModel
MODELS = {
    "mse": {"path": "trained_models/mse",
            "model_cls": RegistrationModel,
            "display_name": "MSE",
            "color": "tab:orange",
            "probability_range": (2, 10),
            "p_tumor_probability_range": (-0.2, 8)},
    "semantic_loss": {"path": "trained_models/semantic_loss",
                      "model_cls": RegistrationModel,
                      "display_name": "Semantic Loss",
                      "color": "tab:blue",
                      "probability_range": (-1960, -500),
                      "p_tumor_probability_range": (0.1, 800)},
    "jac_det_model": {"path": "trained_auxiliary_models/jac_det_model",
                      "model_cls": JacDetModel,
                      "display_name": "Jac. Det.",
                      "color": "tab:green",
                      "probability_range": (0, 10),
                      "p_tumor_probability_range": (-0.05, 0.5)},
    "li_wyatt_model": {"path": "trained_auxiliary_models/li_wyatt_model",
                       "model_cls": LiWyattModel,
                       "display_name": "Li and Wyatt",
                       "color": "tab:red",
                       "probability_range": (0, 1.5),
                       "p_tumor_probability_range": (-0.03, 0.4)},
    "vae_anomaly_detection": {"path": "trained_auxiliary_models/vae_anomaly_detection_model/",
                              "model_cls": VAEAnaomalyDetection,
                              "display_name": "An and Cho",
                              "color": "tab:purple",
                              "probability_range": (2, 10),
                              "p_tumor_probability_range": (-1, 8)},
    "segmentation_model": {"path": "trained_auxiliary_models/segmentation_model",
                           "model_cls": SegmentationModel,
                           "display_name": "Supervised Seg.",
                           "color": "darkgrey",
                           "p_tumor_probability_range": (0, 1)}
}

FULL_MODELS = ["mse", "semantic_loss"]  # only full models of our method
# all models, including baselines
#
ALL_MODELS = ["semantic_loss", "mse", "li_wyatt_model",
              "jac_det_model", "vae_anomaly_detection", "segmentation_model"]
