from tensorflow.keras.utils \
    import set_random_seed, image_dataset_from_directory
from tensorflow.keras.layers import Rescaling

from .config import config


# Pipeline with a single function : bare minimum
def get_pipeline():
    train_ds = image_dataset_from_directory(
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

    val_ds = image_dataset_from_directory(
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

    norm_layer = Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (norm_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (norm_layer(x), y))

    return {
        "train": train_ds,
        "val": val_ds
    }


set_random_seed(config.RANDOM_STATE)
pipeline = get_pipeline()


def pipeline_smoke_test():
    for images, labels in pipeline["train"].take(1):
        print(images.numpy().shape, labels.numpy().shape)


if __name__ == "__main__":
    pipeline_smoke_test()
