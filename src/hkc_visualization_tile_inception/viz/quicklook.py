# FILE: src/hkc_visualization_tile/viz/quicklook.py
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import torch


def save_channel_slice(x: torch.Tensor, ch: int, out_path: str, title: str = "") -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    a = x.detach().float().cpu()
    if a.ndim != 3:
        raise ValueError("Expected [H,W,C]")
    if not (0 <= ch < a.shape[-1]):
        raise ValueError("ch out of range")
    img = a[..., ch].numpy()

    plt.figure(figsize=(6, 5))
    plt.title(title or f"Channel slice ch={ch}")
    plt.imshow(img, aspect="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
