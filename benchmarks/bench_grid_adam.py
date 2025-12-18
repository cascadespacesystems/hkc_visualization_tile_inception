# FILE: benchmarks/bench_grid_adam.py
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List

import torch

from hkc_visualization_tile_inception.optim.baseline import adam_optimize_field


def atomic_write_json(path: str, payload: Any) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_bench_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def resolve_device(ds: str) -> torch.device:
    ds = ds.lower()
    if ds == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if ds == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_target(ny: int, nx: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    t = torch.randn((ny, nx), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    t = 0.25 * t / (t.abs().mean().clamp_min(1e-6))
    return t


def main():
    ap = argparse.ArgumentParser(description="Baseline Adam benchmark (public-safe).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=384)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--target_loss", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--out", default="benchmarks/results/bench_baseline.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    x = torch.zeros((args.ny, args.nx), device=device, dtype=dtype, requires_grad=True)
    target = make_target(args.ny, args.nx, device=device, dtype=dtype, seed=args.seed)

    res = adam_optimize_field(x, target, steps=args.steps, target_loss=args.target_loss)

    payload = {
        "schema": "bench_v1",
        "meta": {
            "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
            "device": str(device),
            "nx": args.nx,
            "ny": args.ny,
            "steps": args.steps,
            "target_loss": args.target_loss,
            "seed": args.seed,
            "dtype": args.dtype,
            "torch_version": getattr(torch, "__version__", "unknown"),
        },
        "runs": [res.__dict__],
    }
    atomic_write_json(args.out, payload)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
