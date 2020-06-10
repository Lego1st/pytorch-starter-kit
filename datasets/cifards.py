import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from albumentations import *
from albumentations.pytorch import ToTensor

class CifarDS(Dataset):
    
    def __init__(self, cfg, mode="train"):
        super(CifarDS, self).__init__()
        self.DATA_DIR = Path(cfg.DIRS.DATA)
        if mode == "test":
            self.df = pd.read_csv(str(self.DATA_DIR / "test.csv"))
        else:   
            self.df = pd.read_csv(str(self.DATA_DIR / "folds" / f"{mode}_fold{cfg.TRAIN.FOLD}.csv")) 
        self.data_root = self.DATA_DIR / ("test" if mode ==  "test" else "train")
        self.mode = mode

        size = cfg.DATA.IMG_SIZE
        self.transform = getTrainTransforms(size) if mode == "train" else getTestTransforms(size)

    def __len__(self):
        return len(self.df)

    def _load_img(self, img_path):
        image = cv2.imread(img_path)
        image = self.transform(image=image)
        image = image["image"]
        return image

    def __getitem__(self, idx):
        info = self.df.loc[idx]
        img_path = str(self.data_root / info["id"])
        image = self._load_img(img_path)
        if self.mode == "test":
            return image    
        else:
            label = info["label"]
            return image, label


def getTrainTransforms(size):

    transforms_train = Compose([
        Resize(size[0], size[1]),
        OneOf([
            RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4),
            RandomGamma(gamma_limit=(50, 150)),
            IAASharpen(),
            IAAEmboss(),
            CLAHE(clip_limit=2),
            NoOp()
        ]),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            MedianBlur(blur_limit=3),
            Blur(blur_limit=3),
        ], p=0.15),
        OneOf([
            RGBShift(),
            HueSaturationValue(),
        ], p=0.05),
        HorizontalFlip(p=0.5),
        Normalize(),
        ToTensor()
    ])
    return transforms_train

def getTestTransforms(size):

    transforms_test = Compose([
        Resize(size[0], size[1]),
        Normalize(),
        ToTensor()
    ])
    return transforms_test