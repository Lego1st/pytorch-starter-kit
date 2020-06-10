# Pytorch starter kit v2.0

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

```bash
.
├── config.py
├── core
│   ├── checkpoint.py
│   ├── data_manipulater.py
│   ├── losses.py
│   ├── lr_scheduler.py
│   ├── metrics.py
│   ├── optimizer.py
│   ├── registry.py
│   └── trainer.py
├── datasets
│   ├── aux_files
│   ├── balanced_sampler.py
│   ├── cifards.py
│   └── data
├── expconfigs
│   ├── exp0.yaml
│   └── exp1.yaml
├── logs
│   ├── exp0.train.log
│   ├── new_exp.test.log
│   ├── new_exp.train.log
│   └── new_exp.valid.log
├── main.py
├── modeling
│   ├── build.py
│   ├── efficient.py
│   ├── __init__.py
│   ├── layers.py
│   ├── ml_heads.py
│   └── resnet.py
├── outputs
├── README.md
├── symlink.sh
├── tools.py
└── weights
    ├── best_exp0.pth
    ├── best_new_exp.pth
    ├── exp0.pth
    └── new_exp.pth
```

```bash
# Examples command line

# Train
python main.py --config expconfigs/exp0.yaml

# Valid, Test
python main.py --config expconfigs/exp0.yaml --load weights/best_exp0_fold0.pth --valid
python main.py --config expconfigs/exp0.yaml --load weights/best_exp0_fold0.pth --test --tta # test with tta mode

```

### Requirements

yacs, apex, timm, torch, torchvision, pandas, albumentations