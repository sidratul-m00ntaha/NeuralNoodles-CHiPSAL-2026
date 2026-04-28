"""
dataset.py
----------
PyTorch Dataset and transforms for the CHiPSAL 2026 meme image modality.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ── Standard ImageNet normalisation ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Augmentation pipeline used during training folds."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform used for validation and test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MemeImageDataset(Dataset):
    """
    Loads meme images from disk given a DataFrame with columns:
        - 'index'  : image filename (e.g. '1234.jpg')
        - 'label'  : integer class label (0 / 1 / 2)

    For the test set, pass a DataFrame with a dummy 'label' column of zeros.

    Parameters
    ----------
    df        : pd.DataFrame  — split dataframe
    img_dir   : str           — path to folder containing the images
    transform : transforms.Compose — torchvision transform pipeline
    """

    def __init__(self, df, img_dir: str, transform: transforms.Compose):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row       = self.df.iloc[idx]
        img_path  = os.path.join(self.img_dir, row["index"])
        label     = int(row["label"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)