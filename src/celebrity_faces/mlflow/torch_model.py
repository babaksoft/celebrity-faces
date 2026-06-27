"""
Contains the LightningModule which holds the neural network structure
and the logic for forward pass, loss calculation, and optimization configuration.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class FaceClassifier(LightningModule):
    """
    The core classification model using CNN features, followed by a linear classifier
    with dropout regularization. Inherits from LightningModule to handle training flow.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3, lr: float = 3e-4):
        super().__init__()
        # Save hyperparameters for easy logging and access (e.g., self.hparams.lr)
        self.save_hyperparameters()

        # Convolutional Features Block
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling for size invariance
        )

        # Fully Connected Classifier Block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """Defines the sequence of data through the model."""
        x = self.features(x)
        return self.classifier(x)

    def _shared_step(self, batch, stage: str):
        """Calculates loss and accuracy for both training and validation steps."""
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Use self.log to log metrics that Lightning automatically tracks
        prefix = "val_" if stage == "val" else ""
        self.log(f"{prefix}loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{prefix}accuracy", acc, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    # --- Lightning Hooks for Training Flow ---

    def training_step(self, batch, batch_idx):
        """Executed every step during the training phase."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Executed every step during the validation phase."""
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        """Tells Lightning which optimizer to use and how to update weights."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
