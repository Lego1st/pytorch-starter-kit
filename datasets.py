import os

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
        if mode == "test":
            self.df = pd.read_csv(os.path.join(cfg.DIRS.DATA, 'test.csv'))
        else:   
            self.df = pd.read_csv(os.path.join(cfg.DIRS.DATA, "folds", f"{mode}_fold{cfg.TRAIN.FOLD}.csv")) 
        self.data_root = os.path.join(cfg.DIRS.DATA, "test" if mode ==  "test" else "train")
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
        img_path = os.path.join(self.data_root, info["id"])
        image = self._load_img(img_path)
        if self.mode == "test":
            return image    
        else:
            label = info["label"]
            return image, label


def getTrainTransforms(size):

    transforms_train = Compose([
        Resize(size, size),
        # OneOf([
        #     ShiftScaleRotate(
        #         shift_limit=0.0625,
        #         scale_limit=0.1,
        #         rotate_limit=30,
        #         border_mode=cv2.BORDER_CONSTANT,
        #         value=0),
        #     GridDistortion(
        #         distort_limit=0.2,
        #         border_mode=cv2.BORDER_CONSTANT,
        #         value=0),
        #     OpticalDistortion(
        #         distort_limit=0.2,
        #         shift_limit=0.15,
        #         border_mode=cv2.BORDER_CONSTANT,
        #         value=0),
        #     NoOp()
        # ]),
        # RandomSizedCrop(
        #     min_max_height=(int(size * 0.75), size),
        #     height=size,
        #     width=size,
        #     p=0.25),
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
        Resize(size, size),
        Normalize(),
        ToTensor()
    ])
    return transforms_test