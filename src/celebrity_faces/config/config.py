from pathlib import Path


RANDOM_STATE = 147

PROJECT_NAME = "celebrity-faces"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"

IMG_ROOT = DATA_PATH / "raw" / "Celebrity Faces Dataset"
LABELS = [
    "Angelina Jolie", "Kate Winslet", "Natalie Portman", "Nicole Kidman", "Sandra Bullock",
    "Brad Pitt", "Johnny Depp", "Leonardo DiCaprio", "Tom Cruise", "Tom Hanks"
]
LABEL_MAP = { LABELS[idx]: idx for idx in range(len(LABELS)) }
NEW_SIZE = (160, 160) # Final image size for preprocessing
