import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .utils_module import _initialize_weights


class ResNet(nn.Module):

    def __init__(self, model_name, num_classes):
        super(ResNet, self).__init__()

        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
        in_features = model.last_linear.in_features
        del model.last_linear
        feature_map = list(model.children())
        feature_map.pop()
        self.backbone = nn.Sequential(*list(feature_map))

        self.fc = nn.Linear(in_features, num_classes)
    
    def features(self, x):
        return self.backbone(x)
    
    def logits(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.logits(x)
        return x