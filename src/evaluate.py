"""
evaluate.py — Evaluation & Inference for Deepfake Detection Model

Usage:
    python src/evaluate.py --eval_all
    python src/evaluate.py --video path/to/video.mp4
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from fusion import FusionModel
from dataset import get_test_loader, get_trainval_loaders

# ── Paths ──────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints/best_model.pt"
TRAINVAL_CSV    = BASE_DIR / "data/processed/trainval_labels.csv"
NT_CSV          = BASE_DIR / "data/processed/nt_labels.csv"
CELEBDF_CSV     = BASE_DIR / "data/processed/celebdf_labels.csv"
RESULTS_DIR     = BASE_DIR / "results"

# ImageNet stats (same as dataset.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Model Loading ──────────────────────────────────────────
def load_model(checkpoint_path: Path, device: torch.device) -> FusionModel:
    model = FusionModel(device=device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to generate a checkpoint."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    best_auc   = ckpt.get("val_auc", float("nan"))
    best_epoch = ckpt.get("epoch", -1) + 1  # stored 0-indexed

    print(f"\n✅  Model loaded from: {checkpoint_path}")
    print(f"   Best Val AUC from training : {best_auc:.4f}  (epoch {best_epoch})")

    return model


# ── Loader evaluation ──────────────────────────────────────
def evaluate_loader(
    model:      FusionModel,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    split_name: str,
) -> dict:
    model.eval()

    all_probs  = []
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            logits, _ = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)

    # Print metrics
    print(f"\n{'='*50}")
    print(f"  Results: {split_name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"{'='*50}")

    # Confusion matrix heatmap
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {split_name}")
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"confusion_{split_name}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Confusion matrix saved → {cm_path}")

    return {
        "split":       split_name,
        "auc":         round(float(auc),       4),
        "accuracy":    round(float(acc),       4),
        "f1":          round(float(f1),        4),
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall),    4),
    }


# ── Full evaluation ────────────────────────────────────────
def run_full_evaluation(model: FusionModel, device: torch.device):
    results = {}

    # 1. Validation split (in-distribution)
    _, val_loader = get_trainval_loaders(
        TRAINVAL_CSV, val_split=0.2, batch_size=8, num_workers=2
    )
    results["val_indistribution"] = evaluate_loader(
        model, val_loader, device, "val_indistribution"
    )

    # 2. NeuralTextures test (unseen fake type)
    if NT_CSV.exists():
        nt_loader = get_test_loader(NT_CSV, batch_size=8, num_workers=2)
        results["neuraltextures_test"] = evaluate_loader(
            model, nt_loader, device, "neuraltextures_test"
        )
    else:
        print(f"\n⚠  NT CSV not found: {NT_CSV}  (skipping)")

    # 3. CelebDF cross-dataset
    if CELEBDF_CSV.exists():
        celebdf_loader = get_test_loader(CELEBDF_CSV, batch_size=8, num_workers=2)
        results["celebdf_crossdataset"] = evaluate_loader(
            model, celebdf_loader, device, "celebdf_crossdataset"
        )
    else:
        print(f"\n⚠  CelebDF CSV not found: {CELEBDF_CSV}  (skipping)")

    # ── Comparison table ───────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  📋  Cross-Split Comparison")
    print(f"{'='*65}")
    header = f"  {'Split':<30}  {'AUC':>6}  {'Accuracy':>9}  {'F1':>6}"
    print(header)
    print("  " + "─" * 58)
    for key, r in results.items():
        print(
            f"  {key:<30}  {r['auc']:>6.4f}  "
            f"{r['accuracy']*100:>8.2f}%  {r['f1']:>6.4f}"
        )
    print(f"{'='*65}")

    # Interpretation
    if "celebdf_crossdataset" in results:
        c_auc = results["celebdf_crossdataset"]["auc"]
        print(f"\n  Cross-dataset AUC of {c_auc:.4f} means your model "
              f"{'✅ generalizes well' if c_auc >= 0.72 else '⚠ may not generalize well'} "
              f"to unseen datasets.")
        if c_auc >= 0.72:
            print("  🏆  Publication-worthy cross-dataset performance!")

    if "neuraltextures_test" in results:
        nt_auc = results["neuraltextures_test"]["auc"]
        print(f"  NeuralTextures AUC of {nt_auc:.4f} tests generalization "
              f"to {'✅ unseen' if nt_auc >= 0.80 else '❗ potentially problematic'} fake type.")

    # Save JSON summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾  Evaluation summary saved → {summary_path}\n")

    return results


# ── Single-video inference ─────────────────────────────────
def predict_video(video_path: str | Path, model: FusionModel, device: torch.device) -> dict:
    import cv2
    from facenet_pytorch import MTCNN

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"⚠  File not found: {video_path}")
        return {}

    FACE_SIZE  = 224
    NUM_FRAMES = 16

    mtcnn = MTCNN(
        image_size=FACE_SIZE, margin=20, keep_all=False,
        device=device, thresholds=[0.5, 0.6, 0.6], post_process=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠  Cannot open video: {video_path}")
        return {}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int) if total > 0 else range(NUM_FRAMES)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()

        if not ret or frame_bgr is None:
            frames.append(np.zeros((3, FACE_SIZE, FACE_SIZE), dtype=np.float32))
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            face_tensor = mtcnn(frame_rgb)
            if face_tensor is not None:
                face_np = np.clip(face_tensor.numpy(), 0, 255).astype(np.float32) / 255.0  # (3,224,224)
            else:
                raise ValueError("No face")
        except Exception:
            h, w = frame_bgr.shape[:2]
            short = min(h, w)
            y0, x0 = (h - short) // 2, (w - short) // 2
            crop = frame_bgr[y0:y0 + short, x0:x0 + short]
            crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (FACE_SIZE, FACE_SIZE))
            face_np = crop.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3,224,224)

        # Normalise
        for c in range(3):
            face_np[c] = (face_np[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        frames.append(face_np)

    cap.release()

    arr = np.stack(frames, axis=0)                          # (16, 3, 224, 224)
    x   = torch.from_numpy(arr).float().unsqueeze(0).to(device)  # (1, 16, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        logits, confidence = model(x)
        pred_class = logits.argmax(dim=1).item()
        conf_pct   = confidence.item() * 100

        # rPPG anomaly score: std of mean RGB signal (higher = more variation = more real)
        raw_sig     = model.rppg_branch.extract_raw_signal(x)  # (1, 16, 3)
        rppg_anomaly = raw_sig.std().item()

    label_str = "FAKE" if pred_class == 1 else "REAL"

    print(f"\n{'='*40}")
    print(f"  Video : {video_path.name}")
    print(f"  Prediction       : {label_str}")
    print(f"  Confidence       : {conf_pct:.2f}%")
    print(f"  rPPG Anomaly Score : {rppg_anomaly:.4f}")
    print(f"    (higher std = more natural variation = more likely REAL)")
    print(f"{'='*40}\n")

    return {
        "prediction":         label_str,
        "confidence":         round(conf_pct, 2),
        "rppg_anomaly_score": round(rppg_anomaly, 4),
    }


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deepfake Detection — Evaluation & Inference"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video file for single-video prediction"
    )
    parser.add_argument(
        "--eval_all", action="store_true",
        help="Run full evaluation on all test splits"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model = load_model(CHECKPOINT_PATH, device)

    if args.video:
        predict_video(args.video, model, device)
    elif args.eval_all:
        run_full_evaluation(model, device)
    else:
        print("\nUsage:")
        print("  python src/evaluate.py --eval_all")
        print("  python src/evaluate.py --video path/to/video.mp4\n")
