from src.baselines.deterministic_registration_model import DeterministicRegistrationModel
import torch
import torchreg


class JacDetModel(DeterministicRegistrationModel):

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

        # calculate the bound
        jac_det = self.jacobian_determinant(transform)
        # pad to image size
        pad_margin = (0, 0, 0, 1, 0, 1) if self.ndims == 2 else (
            0, 1, 0, 1, 0, 1)
        jac_det = torch.nn.functional.pad(jac_det, pad_margin)

        # Tumors are found by strech/contract of the domain. return -p(J | I)
        bound = (torch.log(torch.clamp(jac_det, 10e-6, 10e6)))**2

        return {"bound": bound,
                "transform": transform,
                "morphed": I01, }
