import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

from celebrity_faces.config import config

OPTIMIZER = Adam(learning_rate=3e-4)


def train(model, pipeline, optimizer=OPTIMIZER, epochs=50):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    history = model.fit(
        pipeline.train_ds,
        epochs=epochs,
        validation_data=pipeline.val_ds,
    )

    return history


def plot_learning_curves(history: dict, experiment: str) -> str:
    epochs = len(history["loss"])
    _, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(range(1, epochs + 1), history["accuracy"], "b-", label="Train accuracy")
    ax[0].plot(
        range(1, epochs + 1),
        history["val_accuracy"],
        "r-",
        label="Validation accuracy",
    )
    ax[0].set(xlabel="Epoch", ylabel="Accuracy")
    ax[0].legend()

    ax[1].plot(range(1, epochs + 1), history["loss"], "b-", label="Train loss")
    ax[1].plot(range(1, epochs + 1), history["val_loss"], "r-", label="Validation loss")
    ax[1].set(xlabel="Epoch", ylabel="Loss")
    ax[1].legend()

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # lc stands for learning curves
    path = config.ARTIFACTS_DIR / f"lc_{experiment}.png"
    plt.savefig(path)
    plt.close()
    return path
