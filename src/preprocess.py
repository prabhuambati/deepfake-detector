"""
preprocess.py — Video Preprocessing Pipeline for Deepfake Detection
Extracts 16 evenly-spaced face-cropped frames from each video and saves as .npy

Usage:
    python src/preprocess.py               # Full preprocessing
    python src/preprocess.py --test_run    # Quick test (first 10 videos per folder)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

# ── Reproducibility ────────────────────────────────────────
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root

FOLDER_CONFIG = {
    # (input_path, output_path, label, split)
    "original":        (BASE_DIR / "data/raw/original",       BASE_DIR / "data/processed/trainval",      0, "trainval"),
    "Deepfakes":       (BASE_DIR / "data/raw/Deepfakes",      BASE_DIR / "data/processed/trainval",      1, "trainval"),
    "Face2Face":       (BASE_DIR / "data/raw/Face2Face",      BASE_DIR / "data/processed/trainval",      1, "trainval"),
    "FaceSwap":        (BASE_DIR / "data/raw/FaceSwap",       BASE_DIR / "data/processed/trainval",      1, "trainval"),
    "NeuralTextures":  (BASE_DIR / "data/raw/NeuralTextures", BASE_DIR / "data/processed/nt_test",       1, "nt"),
    "celebdf_real":    (BASE_DIR / "data/celebdf/real",       BASE_DIR / "data/processed/celebdf_test",  0, "celebdf"),
    "celebdf_fake":    (BASE_DIR / "data/celebdf/fake",       BASE_DIR / "data/processed/celebdf_test",  1, "celebdf"),
}

SPLIT_CSV_MAP = {
    "trainval": BASE_DIR / "data/processed/trainval_labels.csv",
    "nt":       BASE_DIR / "data/processed/nt_labels.csv",
    "celebdf":  BASE_DIR / "data/processed/celebdf_labels.csv",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
NUM_FRAMES  = 16
FACE_SIZE   = 224
FAILED_LOG  = BASE_DIR / "data/processed/failed_videos.txt"


# ── MTCNN Setup (lazy init) ────────────────────────────────
_mtcnn = None

def get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        import torch
        from facenet_pytorch import MTCNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [MTCNN] Initializing on device: {device}")
        _mtcnn = MTCNN(
            image_size=FACE_SIZE,
            margin=20,
            keep_all=False,
            device=device,
            thresholds=[0.5, 0.6, 0.6],  # lower = more sensitive face detection
            post_process=False,           # return raw uint8, not normalized tensor
        )
    return _mtcnn


# ── Frame Extraction ───────────────────────────────────────
def center_crop_resize(frame_bgr: np.ndarray, size: int = FACE_SIZE) -> np.ndarray:
    """Fallback: center-crop + resize to (size, size, 3) in RGB float32 [0,255]."""
    h, w = frame_bgr.shape[:2]
    short = min(h, w)
    y0 = (h - short) // 2
    x0 = (w - short) // 2
    crop = frame_bgr[y0:y0 + short, x0:x0 + short]
    crop = cv2.resize(crop, (size, size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)


def extract_frames(video_path: Path, mtcnn) -> np.ndarray | None:
    """
    Opens video, samples NUM_FRAMES evenly, detects face via MTCNN.
    Returns numpy array of shape (16, 3, 224, 224) float32 in [0,255].
    Returns None if video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()

        if not ret or frame_bgr is None:
            # Fallback: use a blank frame
            frames.append(np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.float32))
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── MTCNN face detection ───────────────────────────
        try:
            face_tensor = mtcnn(frame_rgb)  # returns Tensor (3, H, W) or None
            if face_tensor is not None:
                # face_tensor: (3, 224, 224) range [0, 255] float32
                face_np = face_tensor.numpy()          # (3, 224, 224)
                face_np = np.clip(face_np, 0, 255).astype(np.float32)
                face_hwc = face_np.transpose(1, 2, 0)  # (224, 224, 3)
                frames.append(face_hwc)
            else:
                frames.append(center_crop_resize(frame_bgr))
        except Exception:
            frames.append(center_crop_resize(frame_bgr))

    cap.release()

    # Stack: (16, 224, 224, 3) → permute → (16, 3, 224, 224)
    arr = np.stack(frames, axis=0)               # (16, 224, 224, 3)
    arr = arr.transpose(0, 3, 1, 2)             # (16, 3, 224, 224)
    return arr.astype(np.float32)


# ── Per-folder processing ──────────────────────────────────
def process_folder(
    folder_name: str,
    input_dir: Path,
    output_dir: Path,
    label: int,
    split: str,
    test_run: bool,
    failed_log_handle,
) -> list[dict]:
    """Processes one input folder. Returns list of CSV record dicts."""

    if not input_dir.exists():
        print(f"  ⚠  Folder not found, skipping: {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    mtcnn = get_mtcnn()

    videos = sorted([f for f in input_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in VIDEO_EXTS])
    if test_run:
        videos = videos[:10]

    records = []
    n_ok = 0
    n_fail = 0

    print(f"\n{'─'*55}")
    print(f"  📂  {folder_name}  →  {len(videos)} video(s)")
    print(f"{'─'*55}")

    for vid_path in tqdm(videos, desc=f"  {folder_name}", unit="vid"):
        stem = vid_path.stem
        # Prefix with folder name to avoid collisions across folders
        out_name = f"{folder_name}__{stem}.npy"
        out_path = output_dir / out_name

        # Skip already processed
        if out_path.exists():
            records.append({
                "npy_path": str(out_path),
                "label": label,
                "source_folder": folder_name,
            })
            n_ok += 1
            continue

        frames = extract_frames(vid_path, mtcnn)
        if frames is None:
            failed_log_handle.write(f"{vid_path}\n")
            n_fail += 1
            continue

        np.save(str(out_path), frames)
        records.append({
            "npy_path": str(out_path),
            "label": label,
            "source_folder": folder_name,
        })
        n_ok += 1

    print(f"  ✅  Processed: {n_ok}   |   ❌ Failed: {n_fail}")
    return records


# ── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake video preprocessor")
    parser.add_argument(
        "--test_run", action="store_true",
        help="Only process first 10 videos per folder (quick sanity check)"
    )
    args = parser.parse_args()

    if args.test_run:
        print("\n🚧  TEST RUN MODE — processing first 10 videos per folder\n")

    # Ensure output dirs exist
    for p in SPLIT_CSV_MAP.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Per-split record collector
    split_records: dict[str, list[dict]] = {k: [] for k in SPLIT_CSV_MAP}
    total_ok = 0
    total_fail = 0

    with open(FAILED_LOG, "w") as flog:
        for folder_name, (in_dir, out_dir, label, split) in FOLDER_CONFIG.items():
            recs = process_folder(
                folder_name=folder_name,
                input_dir=in_dir,
                output_dir=out_dir,
                label=label,
                split=split,
                test_run=args.test_run,
                failed_log_handle=flog,
            )
            split_records[split].extend(recs)
            total_ok += len(recs)

    # Count failures from log
    with open(FAILED_LOG) as f:
        total_fail = sum(1 for line in f if line.strip())

    # Save CSVs
    print(f"\n{'='*55}")
    print("  💾  Saving CSV label files …")
    for split_key, csv_path in SPLIT_CSV_MAP.items():
        recs = split_records[split_key]
        df = pd.DataFrame(recs, columns=["npy_path", "label", "source_folder"])
        df.to_csv(csv_path, index=False)
        print(f"  {csv_path.name}  →  {len(df)} entries")

    # Final statistics
    print(f"\n{'='*55}")
    print("  📊  FINAL STATISTICS")
    print(f"{'='*55}")
    for split_key, recs in split_records.items():
        df = pd.DataFrame(recs)
        if df.empty:
            print(f"  {split_key:<20}  0 samples")
        else:
            real_n = (df["label"] == 0).sum()
            fake_n = (df["label"] == 1).sum()
            print(f"  {split_key:<20}  Total: {len(df)}  |  Real: {real_n}  |  Fake: {fake_n}")
    print(f"\n  Total Videos Processed : {total_ok}")
    print(f"  Total Failures          : {total_fail}")
    if total_fail > 0:
        print(f"  Failed videos logged to : {FAILED_LOG}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
