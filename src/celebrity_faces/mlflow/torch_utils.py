"""
Holds reusable logic that does not fit into a DataModule or LightningModule,
such as custom logging callbacks and metric visualization utilities.
"""

from pytorch_lightning import Callback


class HistoryCallback(Callback):
    """
    Custom callback to manually store epoch metrics (loss/accuracy) for plotting,
    as standard logging might not be convenient for direct history access later.
    """

    def __init__(self):
        super().__init__()
        self.history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def _append_metric(self, metrics, key: str):
        """Helper to safely extract a metric value from the trainer's callback metrics."""
        value = metrics.get(key)
        if value is None:
            return
        # Handle different tensor/object types passed by Lightning
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            elif hasattr(value, "item"):
                value = value.item()
            self.history[key].append(float(value))
        except (TypeError, ValueError, RuntimeError) as exc:
            print(f"Warning: Could not log metric {key}. Error: {exc}")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self._append_metric(metrics, "loss")
        self._append_metric(metrics, "accuracy")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self._append_metric(metrics, "val_loss")
        self._append_metric(metrics, "val_accuracy")
