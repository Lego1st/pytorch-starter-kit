import os
import sys
import argparse
import logging
import random
import time
import uuid

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from core.lr_scheduler import make_lr_scheduler
from core.optimizer import make_optimizer
from core.trainer import setup_determinism
from core.losses import build_loss_func
from modeling import *
from config import get_cfg_defaults
from tools import train_loop, valid_model, test_model
from datasets.balanced_sampler import class_balanced_sampler
from datasets.cifards import CifarDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
            help="config yaml path")
    parser.add_argument("--load", type=str, default="",
            help="path to model weight")
    parser.add_argument("-ft", "--finetune", action="store_true",
        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
            help="model runing mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
            help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
            help="enable evaluation mode for testset")
    parser.add_argument("--tta", action="store_true",
            help="enable tta infer")

    parser.add_argument("-d", "--debug", action="store_true",
            help="enable debug mode for test")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args


def setup_logging(args, cfg):

    if not os.path.isdir(cfg.DIRS.LOGS):
        os.mkdir(cfg.DIRS.LOGS)
    
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(
        os.path.join(cfg.DIRS.LOGS, f'{cfg.EXP}.{args.mode}.log'), 
        mode='a'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info(f'===============================')
    logging.info(f'\n\nStart with config {cfg}')
    logging.info(f'Command arguments {args}')



def main(args, cfg):

    logging.info(f"=========> {cfg.EXP} <=========")

    # Declare variables
    start_epoch = 0
    best_metric = 0.

    # Create model
    model = build_model(cfg)

    # Load data
    DataSet = CifarDS
    
    train_ds = DataSet(cfg, mode="train")
    valid_ds = DataSet(cfg, mode="valid")
    test_ds = DataSet(cfg, mode="test")
    sampler = None
    shuffle = True

    # Dataloader
    if cfg.DEBUG:
        train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 100))
        valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 20))

    if cfg.DATA.BALANCE:
        train_labels = np.vectorize(train_ds.label_code)(train_ds.df['label'].values)
        sampler = class_balanced_sampler(train_labels, cfg.DATA.NSAMPLE_PER_CLASS)
        shuffle = False

    train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE, shuffle=shuffle, 
                              drop_last=True, num_workers=cfg.SYSTEM.NUM_WORKERS,
                              sampler=sampler)
    valid_loader = DataLoader(valid_ds, cfg.TRAIN.BATCH_SIZE, shuffle=False, 
                              drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    test_loader = DataLoader(test_ds, cfg.TRAIN.BATCH_SIZE, shuffle=False, 
                              drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

    # Define Loss and Optimizer
    train_criterion = build_loss_func(cfg)
    valid_criterion = build_loss_func(cfg)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer, train_loader)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()
        valid_criterion = valid_criterion.cuda()

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            print(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            if not args.finetune:
                print("resuming optimizer ...")
                optimizer.load_state_dict(ckpt.pop('optimizer'))
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            logging.info(f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    if args.mode == "train":
        train_loop(logging.info, cfg, model, \
                train_loader, train_criterion, valid_loader, valid_criterion, \
                optimizer, scheduler, start_epoch, best_metric)
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_loader, valid_criterion, tta=cfg.INFER.TTA)
    else:
        test_model(logging.info, cfg, model, test_loader, tta=cfg.INFER.TTA)

if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.mode != "train":
        cfg.merge_from_list(['INFER.TTA', args.tta])
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 20]
        cfg.merge_from_list(opts)
    cfg.freeze()
    
    for _dir in ["WEIGHTS", "OUTPUTS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])

    setup_logging(args, cfg) 
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)