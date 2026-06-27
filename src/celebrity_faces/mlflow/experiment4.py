"""
Experiment 4: PyTorch Lightning Dropout + Augmentation

We'll do this experiment with the following settings :
- Objective : Rebuild the best previous model in PyTorch with less boilerplate
- Use the best architecture from Experiment 3 (dropout + stronger augmentation)
- Train with Adam for 30 epochs
- Use the same batch size and image size from config
- Train on GPU with PyTorch Lightning
"""

import mlflow
import torch
from pytorch_lightning import Trainer, seed_everything

from celebrity_faces.config import config
from celebrity_faces.mlflow.torch_model import FaceClassifier
from celebrity_faces.mlflow.torch_pipeline import FaceDataModule
from celebrity_faces.mlflow.torch_utils import HistoryCallback
from celebrity_faces.mlflow.utils import plot_learning_curves

# Constants and Config (Assuming config is globally accessible or passed)

# --- Experiment Settings ---
EPOCHS = 30
EXPERIMENT_NAME = "PyTorch Dropout + Augmentation"
ARTIFACT_NAME = "pytorch_dropout_augment"


def run():
    """
    Main function to initialize, train, and track the experiment.
    """

    print("\n==== Experiment 4: PyTorch Dropout + Augmentation ====\n")

    # Pre-flight checks and initialization
    seed_everything(config.RANDOM_SEED, workers=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 4 requires a CUDA-enabled GPU.")

    # MLflow Setup
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.pytorch.autolog()

    # 1. Data Setup (Using the dedicated pipeline module)
    data_module = FaceDataModule(
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
    )
    data_module.setup("fit")

    # 2. Model Setup (Using the dedicated model module)
    model = FaceClassifier(
        num_classes=len(config.LABELS),
        dropout=0.3,
        lr=3e-4,
    )

    # 3. Utility/Callback Setup and Trainer Configuration
    history_cb = HistoryCallback()
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,
        callbacks=[history_cb],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    # 4. Execute Training and Tracking
    with mlflow.start_run(run_name="main") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        print(f"Starting training and logging to MLflow Run ID: {run.info.run_id}")

        # This single call runs the full data flow, model steps, and callback logic
        trainer.fit(model, datamodule=data_module)

        # 5. Post-Run Artifact Logging (Using the utility module)
        art_path = plot_learning_curves(history_cb.history, ARTIFACT_NAME)
        mlflow.log_artifact(local_path=art_path)
        print("Logged learning curves to MLflow.")

    print("\n==== Experiment 4 (PyTorch Dropout + Augmentation) completed. ====")


if __name__ == "__main__":
    run()
