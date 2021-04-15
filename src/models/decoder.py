import torch.nn as nn
import torchreg.nn as tnn


class FixedDecoder(nn.Module):
    def __init__(self, integrate=False):
        super().__init__()
        self.transformer = tnn.SpatialTransformer()
        self.integrate = tnn.FlowIntegration(
            nsteps=2) if integrate else None

    def forward(self, transform, img, seg=None):
        if self.integrate:
            transform = self.integrate(transform)

        morphed = self.transformer(img, transform)

        if seg is not None:
            morphed_seg = self.transformer(seg, transform, mode="nearest")
        else:
            morphed_seg = None

        return morphed, morphed_seg
