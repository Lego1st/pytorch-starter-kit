import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from core.trainer import _initialize_weights
from .build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    """ResNet for this project
    """
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        backbone_name = cfg.MODEL.RESNET.BACKBONE
        num_classes = cfg.MODEL.NUM_CLASSES
        timm_backbone_name = {
            'r18' : "resnet18",
            'r34' : "resnet34",
            'r50' : "resnet50"
        }
        backbone_model = timm.create_model(
            model_name=timm_backbone_name[backbone_name],
            pretrained=cfg.MODEL.IMAGENET_WEIGHT
        )
        self.out_features = backbone_model.fc.in_features
        del backbone_model.fc
        for attr, block in backbone_model.named_children():
            setattr(self, attr, block) 
        if cfg.MODEL.INP_CHANNEL != 3:
            self.conv1 = nn.Conv2d(cfg.MODEL.INP_CHANNEL, 64, 7, 2, 3, bias=False)
            _initialize_weights(self.conv1)
    
    def forward(self, x, aux=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x_4 = self.layer4(x_3)

        x = self.global_pool(x_4).flatten(1)

        if aux:
            return x, x_3
        
        return x