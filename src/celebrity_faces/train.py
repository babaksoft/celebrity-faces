from functools import partial

from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense
from keras.layers import Activation, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import mlflow

from .config import config
from .pipeline import pipeline


def baseline_model(in_shape):
    def_conv2d = partial(
        Conv2D, kernel_size=3, activation="relu",
        padding="same", kernel_initializer="he_normal"
    )

    model = Sequential([
        Input(shape=in_shape),

        # Conv2D block #1
        def_conv2d(filters=32),
        MaxPool2D(),

        # Conv2D block #2
        def_conv2d(filters=64),
        MaxPool2D(),

        # Conv2D block #3
        def_conv2d(filters=128),
        GlobalAveragePooling2D(),

        Dense(units=10, activation="softmax")
    ])

    return model


def train():
    # Prepare MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Minimal Scratch CNN")
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
            config.MODEL_PATH / "keras_ckpt/baseline.weights.h5",
            monitor="val_loss",
            save_best_only=True, save_weights_only=True
        )
        early_stopping_cb = EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=8,
            restore_best_weights=True
        )

        model.fit(
            pipeline["train"], batch_size=config.BATCH_SIZE, epochs=50,
            validation_data=pipeline["val"],
            callbacks=[checkpoint_cb, early_stopping_cb]
        )


if __name__ == "__main__":
    train()
