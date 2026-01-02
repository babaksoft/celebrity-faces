from pathlib import Path

import tensorflow as tf


RANDOM_STATE = 147
PROJECT_NAME = "celebrity-faces"

# Project structure
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"
IMAGE_ROOT = DATA_PATH / "raw" / "Celebrity Faces Dataset"

# Selected classes and encoded labels
LABELS = [
    "Angelina Jolie", "Kate Winslet", "Natalie Portman", "Nicole Kidman", "Sandra Bullock",
    "Brad Pitt", "Johnny Depp", "Leonardo DiCaprio", "Tom Cruise", "Tom Hanks"
]
LABEL_MAP = { LABELS[idx]: idx for idx in range(len(LABELS)) }

# Dataset splitting
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Data processing
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE
