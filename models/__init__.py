
import apex
from apex import amp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

from .resnet import ResNet
from .utils_module import *


def get_model(cfg):
    model = ResNet
    return model(model_name=cfg.TRAIN.MODEL, 
                 num_classes=cfg.TRAIN.NUM_CLASSES)


def test_model(_print, cfg, model, test_loader, tta=False):
    model.eval()
    pass


def valid_model(_print, cfg, model, valid_loader, valid_criterion, tta=False):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            output = model(image)

            loss = valid_criterion(output, target)
            acc = accuracy(output, target)

            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            top1.update(acc[0], image.size(0))          
            
            tbar.set_description("Valid top1: %.3f, loss: %.3f" % (top1.avg, losses.avg))

    _print("Train top1: %.3f, loss: %.3f" % (top1.avg, losses.avg))
    return top1.avg


def train_loop(_print, cfg, model, train_loader, criterion, valid_loader, valid_criterion, optimizer, scheduler, start_epoch, best_metric):
    
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"Epoch {epoch + 1}")

        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()
        tbar = tqdm(train_loader)

        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()

            # calculate loss
            if np.random.uniform() < cfg.DATA.CUTMIX_PROB:
                mixed_x, y_a, y_b, lam = cutmix_data(image, target)
                output = model(mixed_x)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(image)
                loss = criterion(output, target)
                acc = accuracy(output, target)
            
            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS
            
            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                scheduler(optimizer, i, epoch, None) # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            top1.update(acc[0], image.size(0))
            tbar.set_description("Train top1: %.3f, loss: %.3f, learning rate: %.6f" % (top1.avg, losses.avg, optimizer.param_groups[-1]['lr']))

        _print("Train top1: %.3f, loss: %.3f, learning rate: %.6f" % (top1.avg, losses.avg, optimizer.param_groups[-1]['lr']))

        top1 = valid_model(_print, cfg, model, valid_loader, valid_criterion)
        is_best = top1 > best_metric
        best_metric = max(top1, best_metric)
        
        save_checkpoint({
            "epoch": epoch + 1,
            "arch": cfg.EXP,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")