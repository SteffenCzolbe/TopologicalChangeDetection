from src.baselines.deterministic_registration_model import DeterministicRegistrationModel
import torch
import torchreg


class LiWyattModel(DeterministicRegistrationModel):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.gradient_magnitude = torchreg.nn.GradientMagnitude(sigma=6)

    def normalize(self, t):
        t = t - torch.min(t)
        return t / torch.max(t)

    def forward(self, I0: torch.Tensor, I1: torch.Tensor) -> dict:
        """calculates the upper bound on -log p(I1 | I0)

        In addition, a dict with additional information is returned.

        Args:
            I0 (torch.Tensor): [description]
            I1 (torch.Tensor): [description]
            bidir (bool, optional): [description]. Defaults to False.

        Returns:
            Dictionary with various information
        """

        # register the images
        transform, _ = self.encoder(I0, I1)
        # apply the transformation
        I01 = self.transformer(I0, transform)
        morphed = I0

        # augment the images
        if self.hparams.semantic_loss:
            I0 = self.semantic_loss_model.augment_image(I0)
            I1 = self.semantic_loss_model.augment_image(I1)
            I01 = self.transformer(I0, transform)

        # calculate the bound
        K = 2
        p_diff = self.mse(I01, I1)
        gm = self.gradient_magnitude(I1)
        p_gm = 1 - gm * K
        bound = self.normalize(p_gm) * self.normalize(p_diff)

        return {"bound": bound,
                "transform": transform,
                "morphed": morphed, }
