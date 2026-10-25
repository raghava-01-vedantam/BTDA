from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.unet import build_unet


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{path}'")

    model = build_unet()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = tensor.to(device, dtype=torch.float32)
    logits = model(tensor)
    probs = torch.sigmoid(logits)
    prob_map = probs.squeeze().cpu().numpy()
    return prob_map.astype(np.float32)


def threshold_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    return (prob_map >= threshold).astype(np.uint8)
