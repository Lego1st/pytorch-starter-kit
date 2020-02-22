from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "new_exp" # Experiment name
_C.DEBUG = False

_C.INFER = CN()
_C.INFER.TTA = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 8

_C.DIRS = CN()
_C.DIRS.DATA = "./data/"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"

_C.DATA = CN()
_C.DATA.AUGMENT_PROB = 0.5
_C.DATA.MIXUP_PROB = 0.0
_C.DATA.CUTMIX_PROB = 0.0
_C.DATA.INP_CHANNEL = 3
_C.DATA.IMG_SIZE = 32

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1 
_C.OPT.WARMUP_EPOCHS = 2
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0

_C.TRAIN = CN()
_C.TRAIN.FOLD = 0
_C.TRAIN.MODEL = "new_model" # Model name
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.NUM_CLASSES = 10
_C.TRAIN.DROPOUT = 0.0

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