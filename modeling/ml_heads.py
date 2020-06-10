import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import Registry


ML_HEAD_REGISTRY = Registry("ML_HEAD")
ML_HEAD_REGISTRY.__doc__ = """
Registry for metric learning heads.
"""


def build_ml_head(cfg, in_channels):
    head_name = cfg.MODEL.ML_HEAD.NAME
    out_channels = cfg.MODEL.NUM_CLASSES
    s = cfg.MODEL.ML_HEAD.SCALER
    m = cfg.MODEL.ML_HEAD.MARGIN
    num_centers = cfg.MODEL.ML_HEAD.NUM_CENTERS
    head = ML_HEAD_REGISTRY.get(head_name)(
        in_channels, out_channels, s, m, num_centers
    )
    return head


@ML_HEAD_REGISTRY.register()
class ArcFace(nn.Module):
    """
    This module implements ArcFace.
    """

    def __init__(self, in_channels, out_channels, s=32., m=0.5, num_centers=1):
        super(ArcFace, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.m = m
        self.num_centers = num_centers

        self.weight = nn.Parameter(
            torch.Tensor(out_channels * num_centers, in_channels))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) \
               + ', num_centers=' + str(self.num_centers) + ')'

    def forward(self, inputs, labels):
        # cos(theta)
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = cosine.float()
        if self.num_centers > 1:
            cosine = cosine.view(cosine.size(0), self.num_centers, self.out_channels)
            cosine = F.softmax(cosine * self.s, 1) * cosine
            cosine = cosine.sum(1)
        if not self.training:
            return cosine * self.s
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # print(cosine.dtype)
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


@ML_HEAD_REGISTRY.register()
class CosFace(nn.Module):
    """
    This module implements CosFace (https://arxiv.org/pdf/1801.09414.pdf).
    """

    def __init__(self, in_channels, out_channels, s=64., m=0.35, num_centers=1):
        super(CosFace, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.m = m
        self.num_centers = num_centers

        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels * num_centers, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

    def forward(self, inputs, labels):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = cosine.float()
        if self.num_centers > 1:
            cosine = cosine.view(cosine.size(0), self.num_centers, self.out_channels)
            cosine = F.softmax(cosine * self.s, 1) * cosine
            cosine = cosine.sum(1)
        if not self.training:
            return cosine * self.s
        phi = cosine - self.m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output