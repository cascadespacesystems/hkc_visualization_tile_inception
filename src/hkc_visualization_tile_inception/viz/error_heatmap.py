# FILE: src/hkc_visualization_tile/viz/error_heatmap.py
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def save_error_heatmap(err2d: torch.Tensor, out_path: str) -> None:
    """
    err2d: [H, W] tensor on CPU or GPU
    Writes a PNG heatmap.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    a = err2d.detach().float().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.title("Replay Error Heatmap (mean |error| across channels)")
    plt.imshow(a, aspect="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
