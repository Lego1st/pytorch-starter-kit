from yacs.config import CfgNode as CN


_C = CN()

# ------------------------------------------------------------------------------------ #
# Environment options
# ------------------------------------------------------------------------------------ #
_C.EXP = "new_exp" # Experiment name
_C.DEBUG = False
_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 8

# ------------------------------------------------------------------------------------ #
# Directory options
# ------------------------------------------------------------------------------------ #
_C.DIRS = CN()
_C.DIRS.DATA = "./datasets/data/"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"

# ------------------------------------------------------------------------------------ #
# Data options
# ------------------------------------------------------------------------------------ #
_C.DATA = CN()
_C.DATA.AUGMENT_PROB = 0.5
_C.DATA.MIXUP_PROB = 0.0
_C.DATA.CUTMIX_PROB = 0.0
_C.DATA.IS_PREPROCESSED = True
_C.DATA.BALANCE = False
_C.DATA.NSAMPLE_PER_CLASS = 10
_C.DATA.INTER_RESIZE = 2 # 2: Linear, 3: Cubic
_C.DATA.IMG_SIZE = (32, 32)

# ------------------------------------------------------------------------------------ #
# Modeling options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CN()
_C.MODEL.NAME = "ResNet_v0"
_C.MODEL.IMAGENET_WEIGHT = True
_C.MODEL.INP_CHANNEL = 3
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.LOSS_FUNC = "CrossEntropyLoss"

# ------------------------------------------------------------------------------------ #
# Model's head options
# ------------------------------------------------------------------------------------ #
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.AUX = False

# ---------------------------------------------------------------------------- #
# Metric learning options
# ---------------------------------------------------------------------------- #
_C.MODEL.ML_HEAD = CN()
_C.MODEL.ML_HEAD.ENABLE = False
_C.MODEL.ML_HEAD.NAME = "ArcFace"
_C.MODEL.ML_HEAD.SCALER = 16
_C.MODEL.ML_HEAD.MARGIN = 0.1
# Number of centers
_C.MODEL.ML_HEAD.NUM_CENTERS = 1

# ------------------------------------------------------------------------------------ #
# ResNet options
# ------------------------------------------------------------------------------------ #
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.BACKBONE = "r18"

# ------------------------------------------------------------------------------------ #
# EfficientNet options
# ------------------------------------------------------------------------------------ #
_C.MODEL.EFFICIENT = CN()
_C.MODEL.EFFICIENT.BACKBONE = "b0"

# ------------------------------------------------------------------------------------ #
# Optimizer options
# ------------------------------------------------------------------------------------ #
_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.SCHED = "cosine_warmup"
_C.OPT.GD_STEPS = 1 
_C.OPT.WARMUP_EPOCHS = 2
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.ADAM_EPS = 1e-4

# ------------------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------------------ #
_C.TRAIN = CN()
_C.TRAIN.FOLD = 0
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.DROPOUT = 0.0

# ------------------------------------------------------------------------------------ #
# Misc options
# ------------------------------------------------------------------------------------ #
_C.INFER = CN()
_C.INFER.TTA = False
_C.CONST = CN()
_C.CONST.LABELS = [
  "airplane","automobile","bird","cat","deer",
  "dog","frog","horse","ship","truck"
]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`