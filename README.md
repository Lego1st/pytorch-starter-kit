# pytorch-dl-starter
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

### Requirements

yacs, apex, pretrainedmodels, torch, torchvision, pandas, albumentations