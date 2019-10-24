import os
import sys
import argparse
import logging
import random
import time
import uuid

import apex
from apex import amp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset


from config import get_cfg_defaults
from models import train_loop, valid_model, test_model, get_model
from datasets import CifarDS
from lr_scheduler import LR_Scheduler
from utils import *

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
    model = get_model(cfg)

    # Define Loss and Optimizer
    train_criterion = nn.CrossEntropyLoss()
    valid_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), 
                            lr=cfg.OPT.BASE_LR, 
                            weight_decay=cfg.OPT.WEIGHT_DECAY)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()
        valid_criterion = valid_criterion.cuda()
    
    if cfg.SYSTEM.FP16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, 
                                          opt_level=cfg.SYSTEM.OPT_L, 
                                          keep_batchnorm_fp32=(True if cfg.SYSTEM.OPT_L == "O2" else None))

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

    # Load data
    DataSet = CifarDS
    train_ds = DataSet(cfg, mode="train")
    valid_ds = DataSet(cfg, mode="valid")
    test_ds = DataSet(cfg, mode="test")
    
    # Dataloader
    if cfg.DEBUG:
        train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 20))
        valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 10))

    train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, shuffle=True, 
                              drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, shuffle=False, 
                              drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    test_loader = DataLoader(test_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, shuffle=False, 
                              drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)


    scheduler = LR_Scheduler("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS,\
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg.OPT.WARMUP_EPOCHS)


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
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
    cfg.freeze()
    
    for _dir in ["WEIGHTS", "OUTPUTS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])

    setup_logging(args, cfg) 
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)