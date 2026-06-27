"""
Experiment 1: Minimal Baseline

We'll do this experiment with the following settings :
- Objective : First intuition about model capacity on a noisy dataset (not preprocessed for face detection)
- Use two single conv blocks (32+64) with image augmentation (IA), using Adam optimizer, no GPU optimization
- Use a 64-node dense layer head with no Dropout
- Try bigger image size first (200x200) if no OOM error, otherwise try medium image size (160x160)
- Try medium (16) or bigger (32) batch size
- Train for 50 epochs without early stopping (GPU)
"""

import mlflow
from tensorflow.keras.optimizers import Adam

from celebrity_faces.config import config
from celebrity_faces.mlflow.model import baseline
from celebrity_faces.mlflow.pipeline import Pipeline
from celebrity_faces.mlflow.utils import plot_learning_curves, train


def run():
    print("\n==== Experiment 1: Minimal Baseline ====\n")

    # Experiment setup
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="Minimal Baseline")
    with mlflow.start_run(run_name="main") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.tensorflow.autolog()

        pipeline = Pipeline()
        input_shape = config.IMAGE_SIZE + (3,)
        model = baseline(input_shape)
        optimizer = Adam(learning_rate=3e-4)
        epochs = 50

        history = train(model, pipeline, optimizer, epochs)
        art_path = plot_learning_curves(history.history, "minimal_baseline")
        mlflow.log_artifact(local_path=art_path)
        mlflow.end_run()

    print("\n==== Experiment 1 (Minimal Baseline) completed. ====")


if __name__ == "__main__":
    run()
