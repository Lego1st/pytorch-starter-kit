import apex
from apex import amp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

from core.registry import Registry
from core.metrics import AverageMeter, MetricReport
from sklearn.metrics import accuracy_score
from core.data_manipulater import mixup_data, mixup_criterion
from core.checkpoint import save_checkpoint


def test_model(_print, cfg, model, test_loader, tta=False):
    model.eval()
    tbar = tqdm(test_loader)
    np_outputs = []
    with torch.no_grad():
        for i, image in enumerate(tbar):
            image = image.cuda()
            outputs = model(image)
            output = outputs[0]

            # _, top_1 = torch.topk(output, 1) 
            # y_preds.append(top_1.squeeze(1).cpu().numpy())
            np_output = output.argmax(1).cpu().data.numpy()
            np_outputs.append(np_output)
    print(np_outputs[0].shape)
    np_outputs = np.concatenate(np_outputs, 0)
    print(np_outputs.shape)
    # np.save(os.path.join(cfg.DIRS.OUTPUTS, f"test_{cfg.EXP}.npy"), y_preds)
    # return y_preds


def valid_model(_print, cfg, model, valid_loader, valid_criterion, tta=False):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    tbar = tqdm(valid_loader)
    np_outputs, np_targets = [], []

    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()

            outputs = model(image)
            output = outputs[0]    
            loss = valid_criterion(output, target)

            np_target = target.cpu().data.numpy()
            np_output = output.argmax(1).cpu().data.numpy()

            np_outputs.append(np_output)
            np_targets.append(np_target)
            losses.update(loss.item(), image.size(0))
            top1.update(accuracy_score(np_target, np_output) * 100.0, image.size(0))     

    np_outputs = np.hstack(np_outputs)
    np_targets = np.hstack(np_targets)
    metric = accuracy_score(np_outputs, np_targets)

    _print("Valid acc: %.3f, top1: %.3f, loss: %.3f" % (metric, top1.avg, losses.avg))
    return metric


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
                outputs = model(mixed_x)
                output = outputs[0]
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                outputs = model(image)
                output = outputs[0]
                loss = criterion(output, target)

            np_target = target.cpu().data.numpy()
            np_output = output.argmax(1).cpu().data.numpy()
            
            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS
            
            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            top1.update(accuracy_score(np_target, np_output) * 100.0, image.size(0))
            tbar.set_description("Train top1: %.3f, loss: %.3f, learning rate: %.6f" % (top1.avg, losses.avg, optimizer.param_groups[-1]['lr']))

        _print("Train top1: %.3f, loss: %.3f, learning rate: %.6f" % (top1.avg, losses.avg, optimizer.param_groups[-1]['lr']))

        eval_metric = valid_model(_print, cfg, model, valid_loader, valid_criterion)
        is_best = eval_metric > best_metric
        best_metric = max(eval_metric, best_metric)
        
        save_checkpoint({
            "epoch": epoch + 1,
            "arch": cfg.EXP,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")