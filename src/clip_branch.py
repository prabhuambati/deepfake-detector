"""
clip_branch.py — CLIP Visual Encoder Branch for Deepfake Detection

Extracts per-frame embeddings using frozen ViT-B/32 CLIP model,
then computes a mean embedding and a temporal anomaly vector.

Usage (test):
    python src/clip_branch.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class CLIPBranch(nn.Module):
    """
    Frozen CLIP ViT-B/32 visual encoder.

    Input  : (batch, 16, 3, 224, 224)  — 16 normalised frames per video
    Outputs: (mean_embedding, anomaly_vector)
               mean_embedding : (batch, 512)
               anomaly_vector : (batch, 16)  — per-frame cosine distance from mean
    """

    EMBED_DIM = 512

    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load open_clip model (ViT-B/32, pretrained on OpenAI's WIT)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )

        self.model     = model.to(self.device)
        self.preprocess = preprocess   # stored but not used (frames already 224×224 & normalised)
        self.model.eval()

        # Freeze ALL parameters
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 16, 3, 224, 224)

        Returns:
            mean_embedding : (batch, 512)  L2-normalised mean CLIP vector
            anomaly_vector : (batch, 16)   1 - cosine_sim(frame_emb, mean_emb)
        """
        batch_size = x.shape[0]
        num_frames = x.shape[1]

        # Flatten temporal dimension → (batch*16, 3, 224, 224)
        x_flat = x.view(batch_size * num_frames, *x.shape[2:])

        with torch.no_grad():
            embeddings = self.model.encode_image(x_flat)              # (B*16, 512)
            embeddings = F.normalize(embeddings, dim=-1)              # L2-normalise

        # Reshape → (batch, 16, 512)
        embeddings = embeddings.view(batch_size, num_frames, self.EMBED_DIM)

        # ── Mean embedding ─────────────────────────────────
        mean_emb = embeddings.mean(dim=1)                             # (batch, 512)
        mean_emb = F.normalize(mean_emb, dim=-1)                     # L2-normalise

        # ── Anomaly vector ─────────────────────────────────
        # cosine_sim: (batch, 16)  — broadcast mean_emb across frames
        cosine_sim = (embeddings * mean_emb.unsqueeze(1)).sum(dim=-1) # (batch, 16)
        anomaly_vector = 1.0 - cosine_sim                            # (batch, 16)

        return mean_emb, anomaly_vector


# ── Standalone test ─────────────────────────────────────────
def test_clip_branch():
    print("\n" + "=" * 55)
    print("  🧪  CLIPBranch Unit Test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = CLIPBranch().to(device)
    model.eval()

    # Dummy input: 2 clips, each 16 frames of 3×224×224
    dummy = torch.randn(2, 16, 3, 224, 224).to(device)

    with torch.no_grad():
        mean_emb, anomaly_vec = model(dummy)

    print(f"\n  mean_embedding shape : {tuple(mean_emb.shape)}  (expected: (2, 512))")
    print(f"  anomaly_vector shape : {tuple(anomaly_vec.shape)}  (expected: (2, 16))")

    print(f"\n  anomaly_vector for sample 0 (values near 0 = temporally coherent):")
    print(f"  {anomaly_vec[0].tolist()}")

    # Assertions
    assert mean_emb.shape == (2, 512),  f"Expected (2, 512), got {mean_emb.shape}"
    assert anomaly_vec.shape == (2, 16), f"Expected (2, 16), got {anomaly_vec.shape}"

    print("\n  ✅  All assertions passed.\n")


if __name__ == "__main__":
    test_clip_branch()
