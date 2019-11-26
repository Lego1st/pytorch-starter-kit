
import apex
from apex import amp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from pytorch_toolbelt.inference import tta as pytta

from .resnet import ResNet
from .utils_module import *


def get_model(cfg):

    model = ResNet
    return model(model_name=cfg.TRAIN.MODEL, 
                 num_classes=cfg.TRAIN.NUM_CLASSES)


def test_model(_print, cfg, model, test_loader, tta=False):

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    model.eval()
    tbar = tqdm(test_loader)
    y_preds = []
    with torch.no_grad():
        for i, image in enumerate(tbar):
            image = image.cuda()
            output = model(image)

            _, top_1 = torch.topk(output, 1) 
            y_preds.append(top_1.squeeze(1).cpu().numpy())

    y_preds = np.concatenate(y_preds, 0)
    np.save(os.path.join(cfg.DIRS.OUTPUTS, f"test_{cfg.EXP}.npy"), y_preds)
    return y_preds


def valid_model(_print, cfg, model, valid_loader, valid_criterion, tta=False):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    model.eval()
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            output = model(image)

            loss = valid_criterion(output, target)
            acc = accuracy(output, target)

            losses.update(loss.item(), image.size(0))
            top1.update(acc[0], image.size(0))          

    _print("Valid top1: %.3f, loss: %.3f" % (top1.avg, losses.avg))
    return top1.avg.data.cpu().numpy()[0]


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
            if np.random.uniform() < cfg.DATA.MIXUP_PROB:
                mixed_x, y_a, y_b, lam = mixup_data(image, target)
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