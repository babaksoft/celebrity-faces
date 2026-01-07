from functools import partial

from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense
from keras.layers import Activation, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import mlflow

from .config import config
from .pipeline import pipeline


# Debug model to troubleshoot "Overfit Test" failure
def debug_model(in_shape):
    def_conv2d = partial(
        Conv2D, kernel_size=3, padding="same", kernel_initializer="he_normal"
    )

    model = Sequential([
        Input(shape=in_shape),

        # Conv2D block #1
        def_conv2d(filters=32),
        Activation("relu"),
        MaxPool2D(),

        # Conv2D block #2
        def_conv2d(filters=64),
        Activation("relu"),
        MaxPool2D(),

        # Conv2D block #3
        def_conv2d(filters=128),
        Activation("relu"),
        MaxPool2D(),

        # Conv2D block #4
        def_conv2d(filters=256),
        Activation("relu"),
        GlobalAveragePooling2D(),

        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dense(units=10, activation="softmax")
    ])

    return model


def baseline_model(in_shape):
    def_conv2d = partial(
        Conv2D, kernel_size=3, padding="same", kernel_initializer="he_normal"
    )

    model = Sequential([
        Input(shape=in_shape),

        # Conv2D block #1
        def_conv2d(filters=32),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(),

        # Conv2D block #2
        def_conv2d(filters=64),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(),

        # Conv2D block #3
        def_conv2d(filters=128),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(),

        # Conv2D block #4
        def_conv2d(filters=256),
        BatchNormalization(),
        Activation("relu"),
        GlobalAveragePooling2D(),

        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dropout(0.2),
        Dense(units=10, activation="softmax")
    ])

    return model


# Run standard "overfit tiny dataset" test
def overfit_test():
    # Prepare MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Overfit test (minimal)")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="main"):
        input_shape = config.IMAGE_SIZE + (3,)
        small_ds = pipeline.train_ds.take(5)  # 40 images

        model = debug_model(input_shape)
        model.compile(
            optimizer=Adam(0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(small_ds, epochs=30)


def train():
    # Prepare MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Baseline CNN")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="main"):
        input_shape = config.IMAGE_SIZE + (3,)
        model = baseline_model(input_shape)

        optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizer, metrics=["accuracy"]
        )

        # Setup callbacks
        checkpoint_cb = ModelCheckpoint(
            config.MODEL_PATH / "keras_ckpt/base.weights.h5",
            monitor="val_loss",
            save_best_only=True, save_weights_only=True
        )
        early_stopping_cb = EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=8,
            restore_best_weights=True
        )

        model.fit(
            pipeline.train_ds, batch_size=config.BATCH_SIZE, epochs=50,
            validation_data=pipeline.val_ds,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )


if __name__ == "__main__":
    overfit_test()
