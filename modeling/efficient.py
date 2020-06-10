import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from core.trainer import _initialize_weights
from .build import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
class EfficientNet(nn.Module):
    """ResNet for this project
    """
    def __init__(self, cfg):
        super(EfficientNet, self).__init__()
        backbone_name = cfg.MODEL.EFFICIENT.BACKBONE
        num_classes = cfg.MODEL.NUM_CLASSES
        timm_backbone_name = {
            'b0' : "efficientnet_b0",
            'b2' : "efficientnet_b2"
        }
        backbone_model = timm.create_model(
            model_name=timm_backbone_name[backbone_name], 
            pretrained=cfg.MODEL.IMAGENET_WEIGHT
        )
        self.out_features = backbone_model.classifier.in_features
        del backbone_model.classifier
        for attr, block in backbone_model.named_children():
            setattr(self, attr, block) 
        if cfg.MODEL.INP_CHANNEL != 3:
            self.conv_stem = nn.Conv2d(cfg.MODEL.INP_CHANNEL, 32, 3, 2, 1, bias=False)
            _initialize_weights(self.conv_stem)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        return x