import torch


def make_optimizer(cfg, model):
    """
    Create optimizer with per-layer learning rate and weight decay.
    """
    params = []
    for key, value in model.named_parameters():
        lr = cfg.OPT.BASE_LR
        if not value.requires_grad:
            continue
        weight_decay = cfg.OPT.WEIGHT_DECAY
        if "bias" in key:
            weight_decay = cfg.OPT.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.OPT.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(params, lr, eps=cfg.OPT.ADAM_EPS)
    return optimizer