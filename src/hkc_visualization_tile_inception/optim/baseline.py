# FILE: src/hkc_visualization_tile/optim/baseline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class AdamBenchResult:
    mode: str
    device: str
    steps_ran: int
    first_hit_step: int | None
    best_loss: float
    time_s: float
    nan_count: int


def adam_optimize_field(
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    target_loss: float,
    lr: float = 0.15,
) -> AdamBenchResult:
    """
    Simple baseline optimizer for benchmarking.
    """
    assert x.requires_grad
    device = str(x.device)
    opt = torch.optim.Adam([x], lr=lr)

    best = float("inf")
    first_hit = None
    nan_count = 0

    t0 = torch.cuda.Event(enable_timing=True) if x.is_cuda else None
    t1 = torch.cuda.Event(enable_timing=True) if x.is_cuda else None

    import time
    wall0 = time.perf_counter()
    if t0 is not None:
        t0.record()

    for s in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((x - target) ** 2)
        if not torch.isfinite(loss):
            nan_count += 1
            break
        loss.backward()
        opt.step()

        lv = float(loss.item())
        best = min(best, lv)
        if first_hit is None and lv <= target_loss:
            first_hit = s
            break

    if t1 is not None:
        t1.record()
        torch.cuda.synchronize()
    wall1 = time.perf_counter()

    return AdamBenchResult(
        mode="adam",
        device=device.replace("torch.", ""),
        steps_ran=s if "s" in locals() else 0,
        first_hit_step=first_hit,
        best_loss=float(best),
        time_s=float(wall1 - wall0),
        nan_count=int(nan_count),
    )
