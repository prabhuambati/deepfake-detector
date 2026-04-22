"""
verify_setup.py — Setup Verification Script for Deepfake Detection Project
Run with: python verify_setup.py
"""

import sys
import os
from pathlib import Path

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"


def check(label, fn):
    try:
        fn()
        print(f"  {GREEN}✅ {label}{RESET}")
        return True
    except Exception as e:
        print(f"  {RED}❌ {label}  →  {e}{RESET}")
        return False


print(f"\n{BOLD}{CYAN}{'='*55}")
print("  Deepfake Detection — Setup Verification")
print(f"{'='*55}{RESET}\n")

# ── Python version ────────────────────────────────────────
print(f"{BOLD}Python Version:{RESET}")
pv = sys.version_info
print(f"  Python {pv.major}.{pv.minor}.{pv.micro}")
if pv.major < 3 or (pv.major == 3 and pv.minor < 8):
    print(f"  {YELLOW}⚠  Python 3.8+ recommended{RESET}")
print()

# ── Package imports ───────────────────────────────────────
print(f"{BOLD}Package Imports:{RESET}")

check("torch", lambda: __import__("torch"))
check("torchvision", lambda: __import__("torchvision"))
check("facenet_pytorch (facenet-pytorch)", lambda: __import__("facenet_pytorch"))
check("cv2 (opencv-python)", lambda: __import__("cv2"))
check("numpy", lambda: __import__("numpy"))
check("pandas", lambda: __import__("pandas"))
check("tqdm", lambda: __import__("tqdm"))
check("sklearn (scikit-learn)", lambda: __import__("sklearn"))
check("open_clip (open-clip-torch)", lambda: __import__("open_clip"))
check("matplotlib", lambda: __import__("matplotlib"))
check("seaborn", lambda: __import__("seaborn"))
check("PIL (Pillow)", lambda: __import__("PIL"))
print()

# ── CUDA / GPU ────────────────────────────────────────────
print(f"{BOLD}CUDA / GPU:{RESET}")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  {GREEN}✅ CUDA available{RESET}")
        print(f"     GPU : {gpu_name}")
        print(f"     VRAM: {vram:.1f} GB")
    else:
        print(f"  {YELLOW}⚠  CUDA not available — training will use CPU (slow){RESET}")
except Exception as e:
    print(f"  {RED}❌ Could not check CUDA: {e}{RESET}")
print()

# ── Folder check ──────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def count_videos(folder: Path) -> int:
    if not folder.exists():
        return -1
    return sum(1 for f in folder.iterdir()
               if f.is_file() and f.suffix.lower() in VIDEO_EXTS)


folders = {
    "data/raw/original/": Path("data/raw/original"),
    "data/raw/Deepfakes/": Path("data/raw/Deepfakes"),
    "data/raw/Face2Face/": Path("data/raw/Face2Face"),
    "data/raw/FaceSwap/": Path("data/raw/FaceSwap"),
    "data/raw/NeuralTextures/": Path("data/raw/NeuralTextures"),
    "data/celebdf/": Path("data/celebdf"),
}

print(f"{BOLD}Data Folder Status:{RESET}")
all_ok = True
for label, path in folders.items():
    n = count_videos(path)
    if n == -1:
        print(f"  {RED}❌ MISSING  {label}{RESET}")
        all_ok = False
    else:
        icon = GREEN + "✅" if n > 0 else YELLOW + "⚠ "
        print(f"  {icon}{RESET}  {label:<35}  {n} video(s) found")

if not all_ok:
    print(f"\n  {YELLOW}Tip: Run the Kaggle download commands in README to populate data/{RESET}")
print()

# ── Processed data check ──────────────────────────────────
print(f"{BOLD}Processed Data Status:{RESET}")
proc_items = [
    "data/processed/trainval_labels.csv",
    "data/processed/nt_labels.csv",
    "data/processed/celebdf_labels.csv",
    "data/processed/trainval/",
    "data/processed/nt_test/",
    "data/processed/celebdf_test/",
]
for item in proc_items:
    p = Path(item)
    if p.exists():
        if p.is_dir():
            npy_count = sum(1 for f in p.iterdir() if f.suffix == ".npy")
            print(f"  {GREEN}✅{RESET}  {item:<45}  {npy_count} .npy file(s)")
        else:
            print(f"  {GREEN}✅{RESET}  {item}")
    else:
        print(f"  {YELLOW}–{RESET}   {item}  (not yet created — run preprocess.py)")
print()

# ── Final summary ─────────────────────────────────────────
print(f"{BOLD}{CYAN}{'='*55}")
print("  Verification Complete")
print(f"{'='*55}{RESET}\n")
