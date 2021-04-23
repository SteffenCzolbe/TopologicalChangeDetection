import torch.nn as nn
import torchreg.nn as tnn


class FixedDecoder(nn.Module):
    def __init__(self, integrate=False):
        super().__init__()
        self.transformer = tnn.SpatialTransformer()
        self.integrate = tnn.FlowIntegration(
            nsteps=4) if integrate else None

    def forward(self, flow_field, I, seg=None):
        transform = self.get_transform(flow_field, inverse=False)
        morphed, morphed_seg = self.apply_transform(transform, I, S=seg)
        return morphed, morphed_seg

    def get_transform(self, flow_field, inverse=True):
        if self.integrate:
            transform = self.integrate(flow_field)
            if inverse:
                transform_inv = self.integrate(-flow_field)
        else:
            transform = flow_field
            if inverse:
                transform_inv = -flow_field

        if inverse:
            return transform, transform_inv
        else:
            return transform

    def apply_transform(self, transform, I, S=None):
        morphed = self.transformer(I, transform)

        if S is not None:
            morphed_seg = self.transformer(S, transform, mode="nearest")
        else:
            morphed_seg = None

        return morphed, morphed_seg
