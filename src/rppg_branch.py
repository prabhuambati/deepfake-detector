"""
rppg_branch.py — R(2+1)D-based rPPG Feature Extraction Branch

Uses a pretrained R(2+1)D-18 3D CNN with the backbone frozen except
layer4 + custom FC head.  Approximates remote photoplethysmography (rPPG)
by learning spatiotemporal patterns in face videos.

Usage (test):
    python src/rppg_branch.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models.video as video_models


# ── Helper ─────────────────────────────────────────────────
def count_parameters(model: nn.Module):
    """
    Returns (trainable_params, total_params) and prints a formatted summary.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    def fmt(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    print(f"  Parameters — Trainable: {fmt(trainable)} / Total: {fmt(total)}")
    return trainable, total


# ── Module ─────────────────────────────────────────────────
class rPPGBranch(nn.Module):
    """
    Partially-frozen R(2+1)D-18 backbone for rPPG-style temporal feature extraction.

    Frozen  : stem, layer1, layer2, layer3  (pretrained weights fixed)
    Trainable: layer4 + custom fc head

    Input  : (batch, 16, 3, 224, 224)
    Output : (batch, 128)
    """

    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained R(2+1)D-18
        weights = video_models.R2Plus1D_18_Weights.DEFAULT
        self.model = video_models.r2plus1d_18(weights=weights)

        # 1️⃣  Freeze ALL layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # 2️⃣  Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # 3️⃣  Replace fc with custom head (also trainable)
        in_features = self.model.fc.in_features   # 512 for r2plus1d_18
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        # fc is nn.Module so its params are trainable by default after assignment

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 16, 3, 224, 224)

        Returns:
            rppg_features: (batch, 128)
        """
        # R(2+1)D expects (batch, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)   # (batch, 3, 16, 224, 224)
        rppg_features = self.model(x)   # (batch, 128)
        return rppg_features

    # ------------------------------------------------------------------
    def extract_raw_signal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Debugging / visualisation helper.
        Computes mean R, G, B value across spatial dimensions per frame.

        Args:
            x: (batch, 16, 3, 224, 224)

        Returns:
            signal: (batch, 16, 3)  — mean [R, G, B] per frame (approximated rPPG)
        """
        # x: (batch, T, C, H, W) → mean over spatial dims (H, W)
        signal = x.mean(dim=(-2, -1))   # (batch, 16, 3)
        return signal


# ── Standalone test ─────────────────────────────────────────
def test_rppg_branch():
    print("\n" + "=" * 55)
    print("  🧪  rPPGBranch Unit Test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    model = rPPGBranch().to(device)
    model.eval()

    # Dummy input: 2 clips, 16 frames each
    dummy = torch.randn(2, 16, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(dummy)

    print(f"  Output shape         : {tuple(out.shape)}  (expected: (2, 128))")

    # Parameter count
    count_parameters(model)

    # Raw signal
    with torch.no_grad():
        raw_sig = model.extract_raw_signal(dummy)
    print(f"\n  extract_raw_signal() : {tuple(raw_sig.shape)}  (expected: (2, 16, 3))")

    # Assertions
    assert out.shape == (2, 128),        f"Expected (2, 128), got {out.shape}"
    assert raw_sig.shape == (2, 16, 3),  f"Expected (2, 16, 3), got {raw_sig.shape}"

    print("\n  ✅  All assertions passed.\n")


if __name__ == "__main__":
    test_rppg_branch()
