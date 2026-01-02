from tensorflow.keras.utils \
    import set_random_seed, image_dataset_from_directory
from tensorflow.keras.layers import Rescaling, RandomRotation, RandomFlip
from tensorflow.keras.models import Sequential

from .config import config


class Pipeline:
    def __init__(self):
        self._layers = {
            "rescaling": Rescaling(1./255),
            "augmentation": Sequential([
                RandomRotation(0.1),
                RandomFlip("horizontal")
            ])
        }

        self.train_ds = self._train_pipeline()
        self.val_ds = self._val_pipeline()
        self.test_ds = self._test_pipeline()

    def _train_pipeline(self):
        # Load train set with shuffling
        ds = image_dataset_from_directory(
            config.DATA_PATH / "train",
            labels="inferred",
            label_mode="int",
            batch_size=config.BATCH_SIZE,
            image_size=config.IMAGE_SIZE,
            shuffle=True,
            crop_to_aspect_ratio=True,
            data_format="channels_last",
            verbose=False
        )
        self.train_labels = ds.class_names

        # Rescale to achieve more stable convergence
        ds = ds.map(
            lambda x, y: (self._layers["rescaling"](x), y),
            num_parallel_calls=config.AUTOTUNE
        )

        # Prepare for performance optimization BEFORE data augmentation
        ds = ds.cache()
        ds = ds.shuffle(500)

        # Perform data augmentation, ONLY on train set, ONLY during training
        ds = ds.map(
            lambda x, y: (self._layers["augmentation"](x, training=True), y),
            num_parallel_calls=config.AUTOTUNE
        )

        # Allow CPU/GPU cooperation by prefetching
        ds = ds.prefetch(config.AUTOTUNE)

        return ds

    def _val_pipeline(self):
        # Load validation set without shuffling
        ds = image_dataset_from_directory(
            config.DATA_PATH / "validation",
            labels="inferred",
            label_mode="int",
            batch_size=config.BATCH_SIZE,
            image_size=config.IMAGE_SIZE,
            shuffle=False,
            crop_to_aspect_ratio=True,
            data_format="channels_last",
            verbose=False
        )
        self.val_labels = ds.class_names

        # Rescale to achieve more stable convergence
        ds = ds.map(
            lambda x, y: (self._layers["rescaling"](x), y),
            num_parallel_calls=config.AUTOTUNE
        )

        # Optimize performance by caching and prefetching
        ds = ds.cache().prefetch(config.AUTOTUNE)

        return ds

    def _test_pipeline(self):
        # Load test set without shuffling
        ds = image_dataset_from_directory(
            config.DATA_PATH / "test",
            labels="inferred",
            label_mode="int",
            batch_size=config.BATCH_SIZE,
            image_size=config.IMAGE_SIZE,
            shuffle=False,
            crop_to_aspect_ratio=True,
            data_format="channels_last",
            verbose=False
        )
        self.test_labels = ds.class_names

        # Rescale to achieve more stable convergence
        ds = ds.map(
            lambda x, y: (self._layers["rescaling"](x), y),
            num_parallel_calls=config.AUTOTUNE
        )

        # Optimize performance by caching and prefetching
        ds = ds.cache().prefetch(config.AUTOTUNE)

        return ds


set_random_seed(config.RANDOM_STATE)
pipeline = Pipeline()


def pipeline_smoke_test():
    for images, labels in pipeline.train_ds.take(1):
        print(images.numpy().shape, labels.numpy().shape)


if __name__ == "__main__":
    pipeline_smoke_test()
