import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGlobalAvgPool2d(object):
    """
    JIT-ed global average pooling.
    """
    def __init__(self, flatten=True):
        self.flatten = flatten

    def __call__(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h * w).mean(dim=2)
        if self.flatten:
            return x
        else:
            return x.view(n, c, 1, 1)

class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encoding_channels=128):
        super(LocalBlock, self).__init__()
        self.in_channels = in_channels
        self.encoding_channels = encoding_channels

        self.conv_enc = nn.Sequential(
            nn.Conv2d(in_channels, encoding_channels, 1, bias=False),
            nn.BatchNorm2d(encoding_channels))
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.conv_dec = nn.Sequential(
            nn.Conv2d(encoding_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, shallow_features):
        local_descriptors = self.conv_enc(shallow_features)
        reconstructed_features = self.conv_dec(local_descriptors)
        attention_scores = self.attention_net(shallow_features)
        pooled_features = self.avg_pool(
            attention_scores * reconstructed_features)
        pooled_features = torch.flatten(pooled_features, 1)
        logits = self.fc(pooled_features)
        return pooled_features, logits

class OSMEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, num_excitations=2):
        super(OSMEBlock, self).__init__()
        reduction_channels = in_channels // reduction
        self.avg_pool = FastGlobalAvgPool2d(flatten=False)
        excitation_module = nn.Sequential(OrderedDict([
            ('fc1', nn.Conv2d(in_channels, reduction_channels, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(reduction_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2', nn.Conv2d(reduction_channels, in_channels, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(in_channels)),
            ('sigmoid', nn.Sigmoid())
        ]))
        self.excites = nn.ModuleList([excitation_module
            for i in range(num_excitations)])

    def forward(self, x):
        outputs = []
        x_squeeze = self.avg_pool(x)
        for excite in self.excites:
            x_excite = excite(x_squeeze)
            outputs.append(x_excite * x)
        return outputs