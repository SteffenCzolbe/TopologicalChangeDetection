import torch.nn as nn
import torchreg.nn as tnn


class FixedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = tnn.SpatialTransformer()

    def forward(self, transform, img, seg=None):
        morphed = self.transformer(img, transform)
        if seg:
            morphed_seg = self.transformer(seg, transform, mode="nearest")
        return (morphed, morphed_seg) if seg else morphed
