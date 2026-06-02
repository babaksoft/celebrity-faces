import os

import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.utils import image_dataset_from_directory, set_random_seed

from celebrity_faces.config import config


class Pipeline:
    def __init__(self):
        self._rescaling = Rescaling(1.0 / 255)
        self.train_ds = self._train_pipeline()
        self.val_ds = self._val_pipeline()
        self.test_ds = self._test_pipeline()

    def _train_pipeline(self):
        # Load train set with shuffling
        ds = image_dataset_from_directory(
            config.DATA_DIR / "train",
            labels="inferred",
            label_mode="int",
            batch_size=config.BATCH_SIZE,
            image_size=config.IMAGE_SIZE,
            shuffle=True,
            seed=config.RANDOM_SEED,
            crop_to_aspect_ratio=True,
            data_format="channels_last",
            verbose=False,
        )
        self.train_labels = ds.class_names

        # Rescale to achieve more stable convergence
        ds = ds.map(lambda x, y: (self._rescaling(x), y), num_parallel_calls=1)

        return ds

    def _val_pipeline(self):
        # Load validation set without shuffling
        ds = image_dataset_from_directory(
            config.DATA_DIR / "validation",
            labels="inferred",
            label_mode="int",
            batch_size=config.BATCH_SIZE,
            image_size=config.IMAGE_SIZE,
            shuffle=False,
            crop_to_aspect_ratio=True,
            data_format="channels_last",
            verbose=False,
        )
        self.val_labels = ds.class_names

        # Rescale
        ds = ds.map(lambda x, y: (self._rescaling(x), y), num_parallel_calls=1)

        return ds

    def _test_pipeline(self):
        # Load test set without shuffling
        ds = image_dataset_from_directory(
            config.DATA_DIR / "test",
            labels="inferred",
            label_mode="int",
            batch_size=config.BATCH_SIZE,
            image_size=config.IMAGE_SIZE,
            shuffle=False,
            crop_to_aspect_ratio=True,
            data_format="channels_last",
            verbose=False,
        )
        self.test_labels = ds.class_names

        # Rescale
        ds = ds.map(lambda x, y: (self._rescaling(x), y), num_parallel_calls=1)

        return ds


set_random_seed(config.RANDOM_SEED)

# TF determinism (TF 2.13+)
tf.config.experimental.enable_op_determinism()
os.environ["TF_DETERMINISTIC_OPS"] = "1"


def pipeline_smoke_test():
    pipeline = Pipeline()
    for images, labels in pipeline.train_ds.take(1):
        print("\n-------------------------------------------")
        print("Input  (image batch) shape :", images.numpy().shape)
        print("Output (label batch) shape :", labels.numpy().shape)
        print("-------------------------------------------\n")


if __name__ == "__main__":
    pipeline_smoke_test()
