from functools import partial
from tensorflow.keras.layers import (
    Conv2D,
    RandomRotation,
    RandomFlip,
    RandomSharpness,
    MaxPool2D,
    Dense,
    Dropout,
    Input,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Sequential

# Our preferred Conv2D config
def_conv2d = partial(
    Conv2D,
    kernel_size=3,
    padding="same",
    kernel_initializer="he_normal",
    activation="relu",
)

# Normal augmeentation config
augmentation = Sequential([
    RandomRotation(0.1),
    RandomFlip("horizontal"),
])

# Stronger augmeentation config
augmentation_extra = Sequential([
    RandomSharpness(factor=0.4, value_range=[0, 1]), # Slight blur (close to 0.5: No Blur)
    RandomRotation(0.1),
    RandomFlip("horizontal"),
])


def baseline(in_shape):
    return Sequential([
        Input(shape=in_shape),
        augmentation,
        # Conv2D block #1
        def_conv2d(filters=32),
        MaxPool2D(),
        # Conv2D block #2
        def_conv2d(filters=64),
        GlobalAveragePooling2D(),
        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dense(units=10, activation="softmax"),
    ])


def new_baseline(in_shape):
    return Sequential([
        Input(shape=in_shape),
        augmentation,
        # Conv2D block #1
        def_conv2d(filters=32),
        MaxPool2D(),
        # Conv2D block #2
        def_conv2d(filters=64),
        MaxPool2D(),
        # Conv2D block #3
        def_conv2d(filters=128),  # === Model Improvement ===
        GlobalAveragePooling2D(),
        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dense(units=10, activation="softmax"),
    ])


def new_baseline_dropout(in_shape):
    return Sequential([
        Input(shape=in_shape),
        augmentation,
        # Conv2D block #1
        def_conv2d(filters=32),
        MaxPool2D(),
        # Conv2D block #2
        def_conv2d(filters=64),
        MaxPool2D(),
        # Conv2D block #3
        def_conv2d(filters=128),
        GlobalAveragePooling2D(),
        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dropout(0.3),  # === Model Improvement ===
        Dense(units=10, activation="softmax"),
    ])


def new_baseline_augment(in_shape):
    return Sequential([
        Input(shape=in_shape),
        augmentation_extra,  # === Model Improvement ===
        # Conv2D block #1
        def_conv2d(filters=32),
        MaxPool2D(),
        # Conv2D block #2
        def_conv2d(filters=64),
        MaxPool2D(),
        # Conv2D block #3
        def_conv2d(filters=128),
        GlobalAveragePooling2D(),
        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dense(units=10, activation="softmax"),
    ])


def new_baseline_dropout_augment(in_shape):
    return Sequential([
        Input(shape=in_shape),
        augmentation_extra,  # === Model Improvement ===
        # Conv2D block #1
        def_conv2d(filters=32),
        MaxPool2D(),
        # Conv2D block #2
        def_conv2d(filters=64),
        MaxPool2D(),
        # Conv2D block #3
        def_conv2d(filters=128),
        GlobalAveragePooling2D(),
        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dropout(0.3),  # === Model Improvement ===
        Dense(units=10, activation="softmax"),
    ])
