# FILE: tools/generate_tiles.py
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


def smooth2d(x: np.ndarray, iters: int = 6) -> np.ndarray:
    for _ in range(iters):
        up = np.roll(x, -1, axis=0)
        dn = np.roll(x, 1, axis=0)
        lf = np.roll(x, -1, axis=1)
        rt = np.roll(x, 1, axis=1)
        x = 0.2 * (x + up + dn + lf + rt)
    return x


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic .npy tiles for tile-file flow.")
    ap.add_argument("--out_dir", default="artifacts/tiles")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--tile_h", type=int, default=32)
    ap.add_argument("--tile_w", type=int, default=32)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--harder", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed + (999 if args.harder else 0))

    for i in range(args.count):
        x = rng.standard_normal((args.tile_h, args.tile_w, args.channels)).astype(np.float32)
        # smooth per-channel a bit
        for c in range(args.channels):
            x[..., c] = smooth2d(x[..., c], iters=10 if args.harder else 6)
        # normalize
        x = x / (np.mean(np.abs(x)) + 1e-6)

        np.save(os.path.join(args.out_dir, f"tile_{i:03d}.npy"), x)

        # optional "target" (shifted / perturbed)
        tgt = x + 0.05 * rng.standard_normal(x.shape).astype(np.float32)
        np.save(os.path.join(args.out_dir, f"tile_{i:03d}_target.npy"), tgt)

    print(f"[ok] wrote {args.count} tiles (+ targets) to {args.out_dir}")


if __name__ == "__main__":
    main()
