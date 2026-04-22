"""
fusion.py — Multi-Branch Fusion Model for Deepfake Detection

Combines:
  • CLIPBranch    → mean_embedding (512) + anomaly_vector (16)
  • rPPGBranch    → rppg_features (128)
Total input to MLP: 512 + 16 + 128 = 656

A learnable reliability_weight gates the rPPG contribution.

Usage (test):
    python src/fusion.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from clip_branch import CLIPBranch
from rppg_branch import rPPGBranch, count_parameters


class FusionModel(nn.Module):
    """
    End-to-end deepfake detector fusing CLIP semantics, temporal anomaly,
    and rPPG physiological cues.

    Input  : (batch, 16, 3, 224, 224)
    Outputs: (logits, confidence)
               logits     : (batch, 2)  — raw class scores
               confidence : (batch,)    — max softmax probability
    """

    FUSION_DIM = 512 + 16 + 128   # 656

    def __init__(self, device=None):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Sub-branches ───────────────────────────────────
        self.clip_branch  = CLIPBranch().to(self.device)
        self.rppg_branch  = rPPGBranch().to(self.device)

        # ── Reliability weight (learnable scalar) ──────────
        # sigmoid(reliability_weight) ∈ (0, 1) gates rPPG contribution
        self.reliability_weight = nn.Parameter(torch.tensor(0.5))

        # ── Fusion MLP ─────────────────────────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.FUSION_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

        self.to(self.device)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 16, 3, 224, 224) — normalised frame sequences

        Returns:
            logits     : (batch, 2)
            confidence : (batch,)
        """
        x = x.to(self.device)

        # Step 1: CLIP branch
        mean_emb, anomaly_vec = self.clip_branch(x)    # (B,512), (B,16)

        # Step 2: rPPG branch
        rppg_feats = self.rppg_branch(x)               # (B,128)

        # Step 3: Reliability-weighted rPPG
        weight = torch.sigmoid(self.reliability_weight)
        rppg_feats_weighted = rppg_feats * weight       # (B,128)

        # Step 4: Concatenate
        combined = torch.cat(
            [mean_emb, anomaly_vec, rppg_feats_weighted], dim=1
        )                                               # (B, 656)

        # Step 5: Fusion MLP
        logits = self.fusion_mlp(combined)              # (B, 2)

        # Step 6: Confidence
        probs      = torch.softmax(logits, dim=1)       # (B, 2)
        confidence = probs.max(dim=1).values            # (B,)

        return logits, confidence

    # ------------------------------------------------------------------
    def get_trainable_params(self) -> list[dict]:
        """
        Returns parameter groups for AdamW with per-group learning rates.
        Pass the returned list directly to the optimizer constructor.
        """
        return [
            {"params": self.fusion_mlp.parameters(),          "lr": 1e-4},
            {"params": [self.reliability_weight],              "lr": 1e-4},
            {"params": self.rppg_branch.model.layer4.parameters(), "lr": 1e-5},
            {"params": self.rppg_branch.model.fc.parameters(),    "lr": 1e-5},
        ]

    # ------------------------------------------------------------------
    def print_trainable_summary(self):
        """Prints per-module freeze status and total trainable parameter count."""
        print("\n  📋  Trainable Parameter Summary:")
        print(f"  {'Module':<40} {'Status'}")
        print("  " + "─" * 55)

        modules = {
            "clip_branch (entire)": self.clip_branch,
            "rppg_branch.stem":     self.rppg_branch.model.stem,
            "rppg_branch.layer1":   self.rppg_branch.model.layer1,
            "rppg_branch.layer2":   self.rppg_branch.model.layer2,
            "rppg_branch.layer3":   self.rppg_branch.model.layer3,
            "rppg_branch.layer4":   self.rppg_branch.model.layer4,
            "rppg_branch.fc (head)":self.rppg_branch.model.fc,
            "fusion_mlp":           self.fusion_mlp,
        }

        total_trainable = 0
        for name, module in modules.items():
            params     = list(module.parameters())
            trainable  = sum(p.numel() for p in params if p.requires_grad)
            total_p    = sum(p.numel() for p in params)
            is_frozen  = all(not p.requires_grad for p in params)
            status_str = "🔒 FROZEN" if is_frozen else f"✅ TRAINABLE ({trainable:,} params)"
            print(f"  {name:<40} {status_str}")
            total_trainable += trainable

        # reliability_weight
        print(f"  {'reliability_weight (scalar)':<40} ✅ TRAINABLE (1 param)")
        total_trainable += 1

        print(f"\n  Total Trainable Params: {total_trainable:,}")


# ── Standalone test ─────────────────────────────────────────
def test_fusion():
    print("\n" + "=" * 55)
    print("  🧪  FusionModel Unit Test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    model = FusionModel(device=device)
    model.eval()

    dummy = torch.randn(2, 16, 3, 224, 224).to(device)

    with torch.no_grad():
        logits, confidence = model(dummy)

    print(f"  logits shape     : {tuple(logits.shape)}  (expected: (2, 2))")
    print(f"  confidence       : {confidence.tolist()}")

    model.print_trainable_summary()

    # Assertions
    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"

    print("\n  ✅  All assertions passed.\n")


if __name__ == "__main__":
    test_fusion()
