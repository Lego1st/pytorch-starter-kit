import torch
import torch.nn as nn

from core.registry import Registry
from core.trainer import _initialize_weights
from .ml_heads import build_ml_head
from .layers import LocalBlock

MODEL_REGISTRY = Registry("MODEL_TYPE")
MODEL_REGISTRY.__doc__ = """
Registry for Mammo models.
"""

BACKBONE_REGISTRY = Registry("BACKBONE_TYPE")
BACKBONE_REGISTRY.__doc__ = """
Registry for Mammo cls backbone.
"""

def build_model(cfg):
    """Build the whole model architecture
    """
    model_name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(model_name)(cfg)
    return model


@MODEL_REGISTRY.register()
class ResNet_v0(nn.Module):
    """Original ResNet
    """
    def __init__(self, cfg):
        super(ResNet_v0, self).__init__()
        self.backbone = BACKBONE_REGISTRY.get("ResNet")(cfg)
        in_features = self.backbone.out_features
        self.fc = nn.Linear(in_features, cfg.MODEL.NUM_CLASSES)
        _initialize_weights(self.fc)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return [x]

@MODEL_REGISTRY.register()
class Efficient_v0(nn.Module):
    """Original EfficientNet
    """
    def __init__(self, cfg):
        super(Efficient_v0, self).__init__()
        self.backbone = BACKBONE_REGISTRY.get("EfficientNet")(cfg)
        in_features = self.backbone.out_features
        self.fc = nn.Linear(in_features, cfg.MODEL.NUM_CLASSES)
        _initialize_weights(self.fc)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return [x]