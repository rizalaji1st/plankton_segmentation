import torch
from torch import nn
import segmentation_models_pytorch as smp


class UNetModule(nn.Module):
    def __init__(self):
        super(UNetModule, self).__init__()
        self.model = smp.Unet(encoder_name='efficientnet-b3',
                              in_channels=3,
                              classes=1,
                              activation=None)

    def forward(self, images, **kwargs):
        mask_preds = self.model(images)
        mask_preds = torch.sigmoid(mask_preds)
        mask_preds = (mask_preds > 0.5).float()
        return mask_preds
