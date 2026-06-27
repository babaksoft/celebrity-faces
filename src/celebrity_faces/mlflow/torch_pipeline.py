"""
Encapsulates all data handling logic using LightningDataModule.
"""

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from celebrity_faces.config import config


class FaceDataModule(LightningDataModule):
    """
    Handles loading and transforming the image dataset for PyTorch Lightning.
    """

    def __init__(self, batch_size: int, image_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size

        # Define transformations
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomAdjustSharpness(sharpness_factor=0.4, p=1.0),
                transforms.RandomRotation(degrees=36),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage=None):
        """
        Loads the datasets into memory (or caches them). This hook runs before training starts.
        """
        if stage in (None, "fit"):
            print("Setting up Training Data...")
            self.train_dataset = datasets.ImageFolder(
                config.DATA_DIR / "train",
                transform=self.train_transform,
            )
            print("Setting up Validation Data...")
            self.val_dataset = datasets.ImageFolder(
                config.DATA_DIR / "validation",
                transform=self.eval_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
