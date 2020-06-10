import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .registry import Registry


LOSS_REGISTRY = Registry("LOSS_FUNC")
LOSS_REGISTRY.__doc__ = """
Registry for Mammo losses.
"""

# Register Torch losses
LOSS_REGISTRY.register(CrossEntropyLoss)


def build_loss_func(cfg):
    loss_name = cfg.MODEL.LOSS_FUNC
    return LOSS_REGISTRY.get(loss_name)()


@LOSS_REGISTRY.register()
class MeanSquareErrLoss(nn.Module):
    """
    Cast -> MSELoss
    """
    def __init__(self):
        super(MeanSquareErrLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, logit, target):
        loss = self.criterion(logit.view(-1), target.float())
        return loss

@LOSS_REGISTRY.register()
class MeanAbsErrLoss(nn.Module):
    """
    Cast -> MSELoss
    """
    def __init__(self):
        super(MeanAbsErrLoss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, logit, target):
        loss = self.criterion(logit.view(-1), target.float())
        return loss

@LOSS_REGISTRY.register()
class KnowledgeDistillationLoss(nn.Module):
    """
    Reference: https://nervanasystems.github.io/distiller/knowledge_distillation.html.

    Args:
        temperature (float): Temperature value used when calculating soft targets and logits.
    """
    def __init__(self, temperature=4.):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logit, teacher_logit):
        student_prob = F.softmax(student_logit, dim=-1)
        teacher_prob = F.softmax(teacher_prob, dim=-1).log()
        loss = F.kl_div(teacher_prob, student_prob, reduction="batchmean")
        return loss


@LOSS_REGISTRY.register()
class JSDCrossEntropyLoss(nn.Module):
    """
    Jensen-Shannon divergence + Cross-entropy loss.
    """
    def __init__(self, num_splits=3, alpha=12, clean_target_loss=nn.CrossEntropyLoss()):
        super(JSDCrossEntropyLoss, self).__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        self.cross_entropy_loss = clean_target_loss

    def forward(self, logit, target):
        split_size = logit.shape[0] // self.num_splits
        assert split_size * self.num_splits == logit.shape[0]
        logits_split = torch.split(logit, split_size)
        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]
        logp_mixture = torch.clamp(torch.stack(probs).mean(0), 1e-7, 1.).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction="batchmean") for p_split in probs]) / len(probs)
        return loss


@LOSS_REGISTRY.register()
class SigmoidFocalLoss(nn.Module):
    """
    Compute focal loss from
    `'Focal Loss for Dense Object Detection' (https://arxiv.org/pdf/1708.02002.pdf)`.

    Args:
        gamma (float): (default=2.).
        alpha (float): (default=0.25).
    """
    def __init__(self, gamma=2., alpha=0.25):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def _sigmoid_focal_loss_cpu(self, logit, target, gamma, alpha):
        p = torch.sigmoid(logit)
        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)
        return -target * term1 * alpha - (1 - target) * term2 * (1 - alpha)

    def forward(self, logit, target):
        loss = self._sigmoid_focal_loss_cpu(logit, target, self.gamma, self.alpha)
        pos_inds = torch.nonzero(target > 0).squeeze(1)
        N = target.size(0)
        loss = loss.sum() / (pos_inds.numel() + N)
        return loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr


@LOSS_REGISTRY.register()
class SoftmaxFocalLoss(nn.Module):
    """
    Compute the softmax version of focal loss.
    Loss value is normalized by sum of modulating factors.

    Args:
        gamma (float): (default=2.).
    """
    def __init__(self, gamma=2.):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        ce_loss = F.cross_entropy(
            logit, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        modulate = ((1 - p_t) ** self.gamma)
        loss = modulate * ce_loss / modulate.sum()
        return loss.sum()


@LOSS_REGISTRY.register()
class ReducedSoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2., thresh=0.5):
        super(ReducedSoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.thresh = thresh

    def forward(self, logit, target):
        ce_loss = F.cross_entropy(
            logit, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        modulate = ((1 - p_t) ** self.gamma) / (self.thresh ** self.gamma)
        modulate[p_t < self.thresh] = 1.
        loss = modulate * ce_loss / modulate.sum()
        return loss.sum()


@LOSS_REGISTRY.register()
class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Negative log-likelihood loss with label smoothing.

    Args:
        smoothing (float): label smoothing factor (default=0.1).
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, logit, target):
        logprobs = F.log_softmax(logit, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
