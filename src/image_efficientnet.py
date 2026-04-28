"""
image_efficientnet.py
---------------------
Image modality branch: EfficientNet-B0 fine-tuned for 3-class meme
sentiment. 5-fold stratified CV → OOF probability matrices.

Architecture
------------
  EfficientNet-B0 (ImageNet pre-trained)
  └─ Replace final Linear(1280→1000) with Linear(1280→3)
  Training: AdamW lr=1e-4, CrossEntropyLoss, 4 epochs per fold
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold

from .dataset import MemeImageDataset, get_train_transform, get_val_transform


# ── Model definition ─────────────────────────────────────────────────────────

class EfficientNetSentiment(nn.Module):
    """
    EfficientNet-B0 with a replaced classification head for 3-class output.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()


def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs  = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


# ── 5-fold OOF runner ─────────────────────────────────────────────────────────

def run_image_oof(
    train_df,
    test_df,
    train_img_dir: str,
    test_img_dir: str,
    device,
    n_splits: int = 5,
    random_state: int = 42,
    batch_size: int = 16,
    epochs: int = 4,
    lr: float = 1e-4,
):
    """
    Fine-tune EfficientNet-B0 with 5-fold stratified CV.
    Returns OOF and averaged test probability matrices.

    Parameters
    ----------
    train_df       : pd.DataFrame — columns: index, label
    test_df        : pd.DataFrame — columns: index (label col will be set to 0)
    train_img_dir  : path to training images
    test_img_dir   : path to test images
    device         : torch.device
    n_splits       : CV folds
    random_state   : seed
    batch_size     : DataLoader batch size
    epochs         : fine-tuning epochs per fold
    lr             : AdamW learning rate

    Returns
    -------
    oof_probs  : np.ndarray (n_train, 3)
    test_probs : np.ndarray (n_test, 3)
    """
    n_train   = len(train_df)
    n_test    = len(test_df)
    n_classes = 3

    oof_probs  = np.zeros((n_train, n_classes))
    test_probs = np.zeros((n_test, n_classes))

    # Test dataset is constant across folds
    test_ds = MemeImageDataset(
        test_df.assign(label=0), test_img_dir, get_val_transform()
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["label"])
    ):
        print(f"  [EfficientNet] Fold {fold + 1}/{n_splits}")

        train_split = train_df.iloc[train_idx]
        val_split   = train_df.iloc[val_idx]

        train_ds = MemeImageDataset(train_split, train_img_dir,
                                    get_train_transform())
        val_ds   = MemeImageDataset(val_split, train_img_dir,
                                    get_val_transform())

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2)

        model     = EfficientNetSentiment(num_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"    Epoch {epoch + 1}/{epochs} done")

        # OOF predictions
        val_probs = predict_proba(model, val_loader, device)
        oof_probs[val_idx] = val_probs

        # Test predictions (averaged across folds)
        test_probs += predict_proba(model, test_loader, device) / n_splits

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    print("  [EfficientNet] OOF generation complete.")
    return oof_probs, test_probs