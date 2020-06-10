# Herbarium 2020 FGCV7

Starter Kit for Deep Learning projects

## Example case: Cifar-10

### With initial data directory structure

```bash
cifar
├── train
│   ├── {idx}_{class}.png
│   └──  ...
├── test
│   ├── {idx}_{class}.png
│   └──  ...
└── labels.txt
```

### Project structure

```
.
├── config.py # default config for project
├── datasets.py
├── EDA.ipynb
├── expconfigs # config for each experiment
│   └── exp0.yaml
├── InferAndTest.ipynb
├── logs
│   ├── exp0_fold0.test.log
│   ├── exp0_fold0.train.log
│   └── exp0_fold0.valid.log
├── lr_scheduler.py
├── main.py
├── models
│   ├── __init__.py
│   ├── resnet.py # custom model for project
│   └── utils_module.py # utilizations for model
├── outputs # output files such as prediction for test, inference, validation
│   └── test_exp0_fold0.npy
├── README.md
├── utils.py
└── weights
    ├── best_exp0_fold0.pth
    └── exp0_fold0.pth
```

###

```
# Examples

# Train
python main.py --config expconfigs/exp0.yaml

# Valid, Test
python main.py --config expconfigs/exp0.yaml --load weights/best_exp0_fold0.pth --valid
python main.py --config expconfigs/exp0.yaml --load weights/best_exp0_fold0.pth --test --tta # test with tta mode

```

### Requirements

yacs, apex, pretrainedmodels, torch, torchvision, pandas, albumentations, pytorch_toolbelt