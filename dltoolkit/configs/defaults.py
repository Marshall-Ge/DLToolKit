"""Configs."""
import math
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Evaluateing files
# -----------------------------------------------------------------------------
_C.TRAIN_FILE = ""
_C.VAL_FILE = ""
_C.TEST_FILE = ""

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Kill training if loss explodes over this ratio from the previous 5 measurements.
# Only enforced if > 0.0
_C.TRAIN.KILL_LOSS_EXPLOSION_FACTOR = 0.0

# Dataset.
_C.TRAIN.DATASET = "coco"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "coco"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# DataType
_C.DATA.DATA_TYPE = "image" # image / video

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# The path to the metadata directory.
_C.DATA.PATH_TO_METADATA_DIR = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# Input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = 3

# Negative sampling rate. used in 'itm' task.
_C.DATA.NEGATIVE_SAMPLING_RATE = 0.5

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# The std value of the visual raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The mean value of the visual raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

_C.DATA.USE_PCA = False

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0


# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# The layer-wise decay of learning rate. Set to 1. to disable.
_C.SOLVER.LAYER_DECAY = 1.0

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Adam's beta
_C.SOLVER.BETAS = (0.9, 0.999)

# LARS optimizer
_C.SOLVER.LARS_ON = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# The name of the current task (training or inference).;
# "itm" for image-text matching learning
_C.TASK = ""

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "."

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# Model architectures that has one single pathway.
# vitb16 / vitb32 / vitl14
_C.MODEL.ARCH = "vitb16"

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False

# If True, detach the final fc layer from the network, by doing so, only the
# final fc layer will be trained.
_C.MODEL.DETACH_FINAL_FC = False

# If True, frozen batch norm stats during training.
_C.MODEL.FROZEN_BN = False

# If True, AllReduce gradients are compressed to fp16
_C.MODEL.FP16_ALLREDUCE = False

# If True, use static graph for the model.
# This is useful for models that have a fixed structure and do not change
_C.MODEL.STATIC_GRAPH = False


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized. `NUM_SYNC_DEVICES` cannot be larger than number of
# devices per machine; if global sync is desired, set `GLOBAL_SYNC`.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.NUM_SYNC_DEVICES = 1

# Parameter for NaiveSyncBatchNorm. Setting `GLOBAL_SYNC` to True synchronizes
# stats across all devices, across all machines; in this case, `NUM_SYNC_DEVICES`
# must be set to None.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.GLOBAL_SYNC = False

# ---------------------------------------------------------------------------- #
# custom configs
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLIP_ORI_PATH = None
_C.SOLVER.COSINE_RESTART_EPOCH = 0

def assert_and_infer_cfg(cfg):

    pass
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()