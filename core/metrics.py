import os
import shutil

import torch
from sklearn.metrics import classification_report


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricReport(object):
    
    def __init__(self, target_names):
        self.target_names = target_names
        self.labels = [i for i in range(len(target_names))]
        self.metrics = None 
        self.metrics_dict = {}

    def get_metrics(self, output, target):
        self.metrics_dict = classification_report(target, output, labels=self.labels, target_names=self.target_names, output_dict=True)
        self.metrics = classification_report(target, output, labels=self.labels, target_names=self.target_names, output_dict=False, digits=5)

    def macro_f1(self):
        return self.metrics_dict['macro avg']['f1-score']

    def accuracy(self):
        print(self.metrics_dict)
        return self.metrics_dict['accuracy']


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res