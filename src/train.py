"""
train.py — Training Script for Deepfake Detection Model

Features:
  • Mixed-precision training (torch.cuda.amp)
  • AdamW with per-layer learning rates
  • CosineAnnealingLR scheduler
  • Early stopping (patience-based)
  • AUC-ROC, Accuracy, F1 tracking
  • CSV log + loss/AUC curves saved to logs/

Usage:
    python src/train.py
"""

import os
import sys
import time
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from fusion import FusionModel
from dataset import get_trainval_loaders

# ── Performance hint ───────────────────────────────────────
torch.backends.cudnn.benchmark = True

# ── Constants ──────────────────────────────────────────────
BATCH_SIZE             = 8
NUM_EPOCHS             = 20
EARLY_STOPPING_PATIENCE = 5
SEED                   = 42

# Paths (relative to project root)
BASE_DIR         = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH  = BASE_DIR / "checkpoints/best_model.pt"
LOG_PATH         = BASE_DIR / "logs/training_log.csv"
CURVES_PATH      = BASE_DIR / "logs/training_curves.png"
TRAINVAL_CSV     = BASE_DIR / "data/processed/trainval_labels.csv"


# ── Seed ───────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Train one epoch ────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()

    running_loss = 0.0
    all_preds    = []
    all_labels   = []
    all_probs    = []

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for frames, labels in pbar:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits, _ = model(frames)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * frames.size(0)
        probs  = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        preds  = logits.detach().argmax(dim=1).cpu().numpy()
        labs   = labels.cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labs)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(len(loader.dataset), 1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5   # only one class present in batch

    return avg_loss, auc


# ── Validate ───────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_probs    = []
    all_preds    = []
    all_labels   = []

    pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
    with torch.no_grad():
        for frames, labels in pbar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                logits, _ = model(frames)
                loss = criterion(logits, labels)

            running_loss += loss.item() * frames.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(len(loader.dataset), 1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, auc, acc, f1


# ── Checkpoint ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_auc, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "val_auc":              val_auc,
        },
        path,
    )
    print(f"  💾  Checkpoint saved at epoch {epoch + 1} with val AUC: {val_auc:.4f}")


# ── Plot curves ────────────────────────────────────────────
def plot_curves(log_df: pd.DataFrame, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = log_df["epoch"].values + 1   # 1-indexed

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, log_df["train_loss"], label="Train Loss", linewidth=2, color="#E74C3C")
    ax.plot(epochs, log_df["val_loss"],   label="Val Loss",   linewidth=2, color="#3498DB", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # AUC
    ax = axes[1]
    ax.plot(epochs, log_df["train_auc"], label="Train AUC", linewidth=2, color="#E74C3C")
    ax.plot(epochs, log_df["val_auc"],   label="Val AUC",   linewidth=2, color="#3498DB", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  Training curves saved → {save_path}")


# ── Main ───────────────────────────────────────────────────
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  🚀  Deepfake Detection — Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*55}")

    # Directories
    for d in [CHECKPOINT_PATH.parent, LOG_PATH.parent, BASE_DIR / "results"]:
        d.mkdir(parents=True, exist_ok=True)

    # Data
    print("\n🔄  Loading datasets …")
    train_loader, val_loader = get_trainval_loaders(
        TRAINVAL_CSV, val_split=0.2, batch_size=BATCH_SIZE, num_workers=2
    )

    # Model
    print("\n🔧  Initialising FusionModel …")
    model = FusionModel(device=device)

    # Optimizer & scheduler
    param_groups = model.get_trainable_params()
    optimizer    = AdamW(param_groups, weight_decay=1e-4)
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler    = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler       = GradScaler(enabled=device.type == "cuda")

    # Track best
    best_val_auc     = 0.0
    best_epoch       = 0
    patience_counter = 0
    log_records      = []

    # ── Training loop ──────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        t_start = time.time()

        print(f"\n{'='*55}")
        print(f"  EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*55}")

        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_auc, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        # Current reliability weight (interpretable)
        rw_sigmoid = torch.sigmoid(model.reliability_weight).item()
        # Current LR from first param group
        cur_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t_start

        print(f"\n  Train Loss : {train_loss:.4f}  |  Train AUC : {train_auc:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}  |  Val AUC   : {val_auc:.4f}  "
              f"|  Val Acc : {val_acc * 100:.2f}%  |  Val F1 : {val_f1:.4f}")
        print(f"  Reliability Weight (sigmoid) : {rw_sigmoid:.4f}")
        print(f"  Current LR : {cur_lr:.7f}  |  Epoch time : {elapsed:.1f}s")

        log_records.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_auc":  train_auc,
            "val_loss":   val_loss,
            "val_auc":    val_auc,
            "val_acc":    val_acc,
            "val_f1":     val_f1,
        })

        # ── Early stopping ─────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            best_epoch       = epoch + 1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_auc, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(f"  ⏳  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  🛑  Early stopping triggered at epoch {epoch + 1}!")
                break

    # ── Post-training ──────────────────────────────────────
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"\n  📝  Training log saved → {LOG_PATH}")

    plot_curves(log_df, CURVES_PATH)

    print(f"\n{'='*55}")
    print(f"  🏆  Training Complete!")
    print(f"  Best Val AUC : {best_val_auc:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint   : {CHECKPOINT_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
