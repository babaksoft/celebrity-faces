"""
Experiment 3: Regularization

We'll do this experiment with the following settings :
- Objective : Inspecting the effects of regularization and augmentation
- Use three single conv blocks (32+64+128), same as previous experiment
- Only add dropout to head (0.3)
- Only add random sharpness to IA
- Add both dropout and random sharpness
- Train for 30 epochs with Adam (GPU)
"""

import mlflow
from tensorflow.keras.optimizers import Adam

from celebrity_faces.config import config
from celebrity_faces.mlflow.model import (
    new_baseline,
    new_baseline_augment,
    new_baseline_dropout,
    new_baseline_dropout_augment,
)
from celebrity_faces.mlflow.pipeline import Pipeline
from celebrity_faces.mlflow.utils import plot_learning_curves, train

EPOCHS = 30
pipeline = Pipeline()
input_shape = config.IMAGE_SIZE + (3,)


def train_model(model_fn, optimizer, run_name, artifact_name):
    print(f"\n==== Run name: {run_name} ====\n")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.tensorflow.autolog()

        model = model_fn(input_shape)
        history = train(model, pipeline, optimizer, EPOCHS)
        art_path = plot_learning_curves(history, artifact_name)
        mlflow.log_artifact(local_path=art_path)
        mlflow.end_run()


def run():
    print("\n==== Experiment 3: Regularization ====\n")

    # Experiment setup
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="Regularization")

    # First run (Baseline)
    optimizer = Adam(learning_rate=3e-4)
    train_model(new_baseline, optimizer, "Baseline", "new_baseline")

    # Second run (Dropout)
    optimizer = Adam(learning_rate=3e-4)
    train_model(new_baseline_dropout, optimizer, "Dropout", "new_baseline_dropout")

    # Third run (Augmentation)
    optimizer = Adam(learning_rate=3e-4)
    train_model(
        new_baseline_augment,
        optimizer,
        "Augmentation",
        "new_baseline_augment"
    )

    # Fourth run (Dropout + Augmentation)
    optimizer = Adam(learning_rate=3e-4)
    train_model(
        new_baseline_dropout_augment,
        optimizer,
        "Dropout+Augmentation",
        "new_baseline_dropout_augment",
    )
    print("\n==== Experiment 3 (Regularization) completed. ====")


if __name__ == "__main__":
    run()
