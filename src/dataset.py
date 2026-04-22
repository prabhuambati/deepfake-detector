"""
dataset.py — PyTorch Dataset and DataLoader utilities for Deepfake Detection

Classes / Functions:
    DeepfakeDataset         — torch.utils.data.Dataset
    get_trainval_loaders()  — stratified train/val split with WeightedRandomSampler
    get_test_loader()       — generic test loader (nt, celebdf)
    verify_dataset()        — prints batch sanity info

Run directly:
    python src/dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# ImageNet normalisation constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

BASE_DIR = Path(__file__).resolve().parent.parent


# ── Dataset ────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Loads .npy files of shape (16, 3, 224, 224) saved by preprocess.py.

    Args:
        csv_path (str | Path): Path to label CSV with columns
                               [npy_path, label, source_folder].
        augment  (bool):       Apply random horizontal flip + colour jitter.
    """

    def __init__(self, csv_path: str | Path, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        # Colour jitter parameters (applied per-frame independently)
        self._brightness = 0.2
        self._contrast   = 0.2

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frames_raw = np.load(row["npy_path"])          # (16, 3, 224, 224) float32 [0,255]

        # Convert to tensor and scale to [0, 1]
        frames = torch.from_numpy(frames_raw).float() / 255.0  # (16, 3, 224, 224)

        processed_frames = []
        flip = self.augment and torch.rand(1).item() > 0.5   # same flip decision for all frames

        for i in range(frames.shape[0]):
            frame = frames[i]   # (3, 224, 224)

            # ── Augmentation ──────────────────────────────
            if self.augment:
                # Independent brightness / contrast jitter per frame
                brightness_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self._brightness
                contrast_factor   = 1.0 + (torch.rand(1).item() * 2 - 1) * self._contrast
                frame = TF.adjust_brightness(frame, brightness_factor)
                frame = TF.adjust_contrast(frame, contrast_factor)

                if flip:
                    frame = TF.hflip(frame)

            # ── Normalisation (always) ────────────────────
            frame = TF.normalize(frame, IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())

            processed_frames.append(frame)

        frames_tensor = torch.stack(processed_frames, dim=0)       # (16, 3, 224, 224)
        label_tensor  = torch.tensor(int(row["label"]), dtype=torch.long)

        return frames_tensor, label_tensor


# ── DataLoader helpers ─────────────────────────────────────

def get_trainval_loaders(
    csv_path: str | Path,
    val_split: float = 0.2,
    batch_size: int = 8,
    num_workers: int = 2,
):
    """
    Stratified 80/20 train/val split with WeightedRandomSampler for training.

    Returns:
        train_loader, val_loader
    """
    df = pd.read_csv(csv_path)
    labels = df["label"].values

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # ── Class distribution report ──────────────────────────
    print("\n📊  Class Distribution after Split:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
        real_n = (split_df["label"] == 0).sum()
        fake_n = (split_df["label"] == 1).sum()
        print(f"   {split_name:<6}: Total={len(split_df)}  Real={real_n}  Fake={fake_n}")

    # ── Temporary CSVs for sub-datasets ───────────────────
    import tempfile, os
    train_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    val_csv   = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    train_df.to_csv(train_csv.name, index=False)
    val_df.to_csv(val_csv.name,     index=False)
    train_csv.close()
    val_csv.close()

    train_dataset = DeepfakeDataset(train_csv.name, augment=True)
    val_dataset   = DeepfakeDataset(val_csv.name,   augment=False)

    # ── WeightedRandomSampler ──────────────────────────────
    class_counts = np.bincount(train_df["label"].values)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[train_df["label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Clean up temp files
    os.unlink(train_csv.name)
    os.unlink(val_csv.name)

    return train_loader, val_loader


def get_test_loader(
    csv_path: str | Path,
    batch_size: int = 8,
    num_workers: int = 2,
) -> DataLoader:
    """
    Generic test loader — no shuffle, no augmentation.
    Works for both nt_labels.csv and celebdf_labels.csv.
    """
    dataset = DeepfakeDataset(csv_path, augment=False)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


# ── Sanity check ───────────────────────────────────────────

def verify_dataset(loader: DataLoader, name: str):
    """Loads one batch and prints shape / label distribution / pixel range."""
    print(f"\n{'─'*50}")
    print(f"  🔍  verify_dataset: {name}")
    print(f"{'─'*50}")
    try:
        frames, labels = next(iter(loader))
        print(f"  Batch shape   : {tuple(frames.shape)}")
        print(f"  Label tensor  : {labels.tolist()}")
        real_n = (labels == 0).sum().item()
        fake_n = (labels == 1).sum().item()
        print(f"  Label dist    : Real={real_n}  Fake={fake_n}")
        print(f"  Pixel range   : min={frames.min():.4f}  max={frames.max():.4f}")
        print(f"  Dtype         : {frames.dtype}")
    except StopIteration:
        print("  ⚠  Loader is empty!")


# ── Script entry ───────────────────────────────────────────
if __name__ == "__main__":
    TRAINVAL_CSV = BASE_DIR / "data/processed/trainval_labels.csv"
    NT_CSV       = BASE_DIR / "data/processed/nt_labels.csv"
    CELEBDF_CSV  = BASE_DIR / "data/processed/celebdf_labels.csv"

    missing = [p for p in [TRAINVAL_CSV, NT_CSV, CELEBDF_CSV] if not p.exists()]
    if missing:
        print(f"\n⚠  Missing CSV files (run preprocess.py first):")
        for m in missing:
            print(f"   {m}")
        sys.exit(0)

    print("\n🔄  Loading train/val loaders …")
    train_loader, val_loader = get_trainval_loaders(TRAINVAL_CSV, batch_size=4, num_workers=0)

    print("\n🔄  Loading NeuralTextures test loader …")
    nt_loader = get_test_loader(NT_CSV, batch_size=4, num_workers=0)

    print("\n🔄  Loading CelebDF test loader …")
    celebdf_loader = get_test_loader(CELEBDF_CSV, batch_size=4, num_workers=0)

    verify_dataset(train_loader,   "Train")
    verify_dataset(val_loader,     "Validation")
    verify_dataset(nt_loader,      "NeuralTextures Test")
    verify_dataset(celebdf_loader, "CelebDF Test")

    print("\n✅  dataset.py sanity check complete.\n")
