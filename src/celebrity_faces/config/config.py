from pathlib import Path

# Global config
MLFLOW_TRACKING_URI = "http://localhost:5000"
RANDOM_SEED = 147
PROJECT_NAME = "celebrity-faces"

# Project structure
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "model"
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
METRICS_DIR = PACKAGE_ROOT / "metrics"
IMAGE_ROOT = DATA_DIR / "raw" / "Celebrity Faces Dataset"

# Selected classes
LABELS = [
    "Angelina Jolie",
    "Kate Winslet",
    "Natalie Portman",
    "Nicole Kidman",
    "Sandra Bullock",
    "Brad Pitt",
    "Johnny Depp",
    "Leonardo DiCaprio",
    "Tom Cruise",
    "Tom Hanks",
]

# Dataset splitting
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Data processing
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 16
