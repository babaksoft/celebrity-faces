# Celebrity Faces - Robust Image Classification Under Limited Data

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://github.com/babaksoft/celebrity-faces/raw/refs/heads/master/pyproject.toml)
![Static Badge](https://img.shields.io/badge/task-classification-orange)
![Static Badge](https://img.shields.io/badge/framework-tensorflow-orange)
![Static Badge](https://img.shields.io/badge/framework-pytorch-orange)
![GitHub License](https://img.shields.io/github/license/babaksoft/celebrity-faces)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/babaksoft/celebrity-faces/build.yml)


## Project Overview

This project investigates how modern convolutional neural networks behave when trained on a small, unconstrained facial image dataset.

Unlike curated benchmark datasets where faces are tightly aligned and cropped, this dataset contains real-world celebrity photographs with significant variation in

- viewing angle
- facial pose
- facial expression
- lighting
- image resolution
- background clutter
- distance from the camera
- partial body visibility

The project intentionally starts with CNNs trained from scratch before transitioning to transfer learning.

Rather than immediately applying pretrained models, the goal is to understand the limitations of feature learning under realistic data constraints.

---

## Simulated Business Problem

Imagine a media company wants to automatically tag celebrity photographs uploaded by journalists.

The system should

- identify one of 10 celebrities
- operate on unconstrained photographs
- tolerate varying zoom levels
- tolerate cluttered backgrounds
- require minimal manual preprocessing

The available labeled dataset is small (approximately 800 training images), making transfer learning a likely necessity.

The engineering question therefore becomes

> How far can a CNN trained from scratch go before pretrained feature extractors become necessary?

---

## Dataset

- Kaggle Celebrity Faces Dataset
- 10 selected celebrities
- 100 images per class
- train / validation / test split

Characteristics

- unaligned faces
- varying scales
- different poses
- upper-body and full-body photographs
- inconsistent backgrounds

These characteristics intentionally make the task more realistic than typical benchmark datasets.

---

## Objectives

- Build reproducible TensorFlow pipelines
- Design CNNs from scratch
- Compare optimizers
- Investigate regularization
- Study augmentation effects
- Compare TensorFlow and PyTorch implementations
- Justify transition to transfer learning
- Track experiments using MLflow
- Version datasets with DVC

---

## Technology Stack

- TensorFlow / Keras
- PyTorch
- PyTorch Lightning
- MLflow
- DVC
- NumPy
- Matplotlib

---

## Key Findings

Training from scratch consistently improved training performance but showed limited validation improvements, suggesting that the primary limitation was feature representation rather than optimization.

These findings motivated the transition to pretrained ImageNet feature extractors.
