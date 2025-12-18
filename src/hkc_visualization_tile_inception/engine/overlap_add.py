# FILE: src/hkc_visualization_tile/engine/overlap_add.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class OverlapAddPlan:
    global_h: int
    global_w: int
    tile_h: int
    tile_w: int
    overlap: int

    @property
    def stride_h(self) -> int:
        return max(1, self.tile_h - self.overlap)

    @property
    def stride_w(self) -> int:
        return max(1, self.tile_w - self.overlap)


def overlap_add_accumulate(
    acc: torch.Tensor,
    wacc: torch.Tensor,
    tile: torch.Tensor,
    top: int,
    left: int,
) -> None:
    """
    acc:  [H, W, C]
    wacc: [H, W, 1]
    tile: [th, tw, C]
    """
    th, tw, c = tile.shape
    H, W, C = acc.shape
    assert c == C

    y0, y1 = top, min(top + th, H)
    x0, x1 = left, min(left + tw, W)

    ty1 = y1 - y0
    tx1 = x1 - x0
    if ty1 <= 0 or tx1 <= 0:
        return

    acc[y0:y1, x0:x1, :] += tile[:ty1, :tx1, :]
    wacc[y0:y1, x0:x1, 0] += 1.0


def overlap_add_finalize(acc: torch.Tensor, wacc: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalizes overlap-add accumulator by weights.
    """
    return acc / (wacc.clamp_min(eps))
