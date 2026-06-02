"""
Experiment 2: New Baseline

We'll do this experiment with the following settings :
- Objective : Inspecting the effects of model capacity vs. optimization
- Use three single conv blocks (32+64+128) with image augmentation (IA), using different optimizers, no GPU optimization
- Use a 64-node dense layer head with no Dropout
- Try bigger image size first (200x200) if no OOM error, otherwise try medium image size (160x160)
- Try medium (16) or bigger (32) batch size
- Train for 30 epochs without early stopping (GPU)
"""

import mlflow
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from celebrity_faces.config import config
from celebrity_faces.mlflow.model import new_baseline
from celebrity_faces.mlflow.pipeline import Pipeline
from celebrity_faces.mlflow.utils import plot_learning_curves, train

EPOCHS = 30
pipeline = Pipeline()
input_shape = config.IMAGE_SIZE + (3,)


def train_model(optimizer, run_name):
    print(f"\n==== Run name: {run_name} ====\n")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.tensorflow.autolog()

        model = new_baseline(input_shape)
        epochs = EPOCHS

        history = train(model, pipeline, optimizer, epochs)
        art_path = plot_learning_curves(history, f"new_baseline_{run_name.lower()}")
        mlflow.log_artifact(local_path=art_path)
        mlflow.end_run()


def run():
    print("\n==== Experiment 2: New Baseline ====\n")

    # Experiment setup
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="New Baseline")

    # First run (Adam)
    optimizer = Adam(learning_rate=3e-4)
    train_model(optimizer, "Adam")

    # Second run (RMSprop)
    optimizer = RMSprop(learning_rate=3e-4)
    train_model(optimizer, "RMSprop")

    # Third run (SGD)
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    train_model(optimizer, "SGD")
    print("\n==== Experiment 2 (New Baseline) completed. ====")


if __name__ == "__main__":
    run()
