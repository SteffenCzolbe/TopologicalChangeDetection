from src.registration_model import RegistrationModel
from src.baselines.vae_anomaly_detection import VAEAnaomalyDetection
from src.baselines.jac_det_model import JacDetModel
from src.baselines.li_wyatt_model import LiWyattModel
from src.baselines.segmentation_model import SegmentationModel
MODELS = {
    "mse": {"path": {"brain2d": "weights/brain2d/topology_detection/mse/weight_decay0.0005",
                     "platelet-em": "weights/platelet-em/topology_detection/mse/weight_decay0.01"},
            "model_cls": RegistrationModel,
            "display_name": "MSE",
            "color": "tab:orange",
            # probability range for heatmap. Use src/eval/overlay.py to finetune
            "probability_range": {"brain2d": (2, 10),
                                  "platelet-em": (3, 7)},
            "p_tumor_probability_range": (-0.2, 6)},
    "semantic_loss": {"path": {"brain2d": "weights/brain2d/topology_detection/semantic_loss/weight_decay0.0005",
                               "platelet-em": "weights/platelet-em/topology_detection/semantic_loss/weight_decay0.01"},
                      "model_cls": RegistrationModel,
                      "display_name": "Semantic Loss",
                      "color": "tab:blue",
                      "probability_range": {"brain2d": (-70, 160),
                                            "platelet-em": (-85, -55)},
                      "p_tumor_probability_range": (-5, 120)},
    "jac_det_model": {"path": {"brain2d": "weights/brain2d/baselines/jac_det_model",
                               "platelet-em": "weights/platelet-em/baselines/jac_det_model"},
                      "model_cls": JacDetModel,
                      "display_name": "Jac. Det.",
                      "color": "tab:green",
                      "probability_range": {"brain2d": (0, 10),
                                            "platelet-em": ()},
                      "p_tumor_probability_range": (-0.05, 0.5)},
    "li_wyatt_model": {"path": {"brain2d": "weights/brain2d/baselines/li_wyatt_model",
                                "platelet-em": "weights/platelet-em/baselines/li_wyatt_model"},
                       "model_cls": LiWyattModel,
                       "display_name": "Li and Wyatt",
                       "color": "tab:red",
                       "probability_range":  {"brain2d": (0, 1.5),
                                              "platelet-em": ()},
                       "p_tumor_probability_range": (-0.03, 0.4)},
    "vae_anomaly_detection": {"path": {"brain2d": "weights/brain2d/baselines/vae_anomaly_detection_model/",
                                       "platelet-em": "weights/platelet-em/baselines/vae_anomaly_detection_model/"},
                              "model_cls": VAEAnaomalyDetection,
                              "display_name": "An and Cho",
                              "color": "tab:purple",
                              "probability_range":  {"brain2d": (2, 10),
                                                     "platelet-em": ()},
                              "p_tumor_probability_range": (-1, 8)},
    "segmentation_model": {"path": {"brain2d": "weights/brain2d/baselines/segmentation",
                                    "platelet-em": "weights/platelet-em/baselines/segmentation"},
                           "model_cls": SegmentationModel,
                           "display_name": "Supervised Seg.",
                           "color": "darkgrey",
                           "p_tumor_probability_range": (0, 1)}
}

#
FULL_MODELS = ["mse", "semantic_loss"]  # only full models of our method
# all models, including baselines
#
ALL_MODELS = ["semantic_loss", "mse", "li_wyatt_model",
              "jac_det_model", "vae_anomaly_detection", "segmentation_model"]
