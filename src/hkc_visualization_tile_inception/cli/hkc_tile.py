# FILE: src/hkc_visualization_tile_inception/cli/hkc_tile.py
from __future__ import annotations

import argparse
import os
import shutil

import torch

# NEW (correct)
from hkc_visualization_tile_inception.engine.spectral_tile_field import (
    SpectralTileFieldConfig,
    make_synthetic_tiles,
    build_global_from_tiles,
    export_tile_coeffs_and_manifest,
    rebuild_global_from_coeffs,
)
from hkc_visualization_tile_inception.engine.validate_recon import validate_reconstruction
from hkc_visualization_tile_inception.io.tile_file_io import save_pt


DEFAULT_OUT = "artifacts/harmonic_export"


def cmd_build_global(args: argparse.Namespace) -> None:
    cfg = SpectralTileFieldConfig(
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        global_h=args.global_h,
        global_w=args.global_w,
        overlap=args.overlap,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )

    tiles = make_synthetic_tiles(cfg)
    X, wacc, placements = build_global_from_tiles(cfg, tiles)

    os.makedirs(args.out_dir, exist_ok=True)
    global_path = os.path.join(args.out_dir, f"global_{cfg.global_h}x{cfg.global_w}x{cfg.out_channels}.pt")
    weight_path = os.path.join(args.out_dir, "global_weight_acc.pt")

    save_pt(global_path, X.detach().cpu())
    save_pt(weight_path, wacc.detach().cpu())

    export_tile_coeffs_and_manifest(cfg, X, placements, out_dir=args.out_dir, keep_frac=args.keep_frac)

    print("=== build-global summary ===")
    print(f"device: {cfg.device}")
    print(f"global_path: {global_path}")
    print(f"weight_path: {weight_path}")
    print(f"manifest: {os.path.join(args.out_dir, 'manifest_tiles.json')}")
    print(f"coeff_dir: {os.path.join(args.out_dir, 'tile_coeffs')}")


def cmd_validate_recon(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    manifest = os.path.join(out_dir, "manifest_tiles.json")

    # Find global tensor by pattern
    globals_ = [p for p in os.listdir(out_dir) if p.startswith("global_") and p.endswith(".pt")]
    if not globals_:
        raise SystemExit("[error] no global_*.pt found. Run build-global first.")
    global_path = os.path.join(out_dir, sorted(globals_)[-1])
    weight_path = os.path.join(out_dir, "global_weight_acc.pt")
    if not os.path.exists(weight_path):
        raise SystemExit("[error] missing global_weight_acc.pt. Run build-global first.")

    # Optional "replay-only" mode: delete global files then rebuild from coeffs
    if args.replay_only:
        backup_global = os.path.join(out_dir, "_backup_" + os.path.basename(global_path))
        backup_weight = os.path.join(out_dir, "_backup_global_weight_acc.pt")

        # Backup what exists (robust to partial runs)
        if os.path.exists(global_path):
            shutil.copy2(global_path, backup_global)
        else:
            raise SystemExit(f"[error] replay_only requested but missing global: {global_path}")

        if os.path.exists(weight_path):
            shutil.copy2(weight_path, backup_weight)
        else:
            # Not fatal: we can still rebuild and validate against the backed-up global.
            backup_weight = None

        # Delete originals (only if present)
        if os.path.exists(global_path):
            os.remove(global_path)
        if os.path.exists(weight_path):
            os.remove(weight_path)


        # rebuild and write new global
        X_rec, wacc_rec, meta = rebuild_global_from_coeffs(
            manifest_path=manifest,
            out_dir=out_dir,
            device=args.device,
            dtype=args.dtype,
        )
        global_path = os.path.join(out_dir, os.path.basename(backup_global).replace("_backup_", ""))
        save_pt(global_path, X_rec.detach().cpu())
        save_pt(weight_path, wacc_rec.detach().cpu())

        # validate against backup originals
        saved_global_path = backup_global
        saved_weight_path = backup_weight if backup_weight is not None else weight_path
        print("[replay-only] rebuilt global from coeffs + manifest.")
    else:
        saved_global_path = global_path
        saved_weight_path = weight_path

    fig_path = args.fig_path
    summary = validate_reconstruction(
        manifest_path=manifest,
        coeff_root=out_dir,
        saved_global_path=saved_global_path,
        saved_weight_path=saved_weight_path,
        device=args.device,
        dtype=args.dtype,
        out_fig_path=fig_path,
    )

    print("=== validate-recon summary ===")
    print(f"tiles_total_in_manifest: {summary.tiles_total}")
    print(f"tiles_used: {summary.tiles_used}")
    print(f"tiles_missing_coeff_path: {summary.tiles_missing}")
    print(f"coverage_frac: {summary.coverage_frac}")
    print(f"mae: {summary.mae}")
    print(f"rmse: {summary.rmse}")
    print(f"maxe: {summary.maxe}")
    print("paths:")
    for k, v in summary.paths.items():
        print(f"  {k}: {v}")
    if fig_path:
        print(f"figure: {fig_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hkc-tile", description="HKC Visualization Tile (public demo CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-global", help="Build global field, export coeff tiles + manifest")
    b.add_argument("--out_dir", default=DEFAULT_OUT)
    b.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    b.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--tile_h", type=int, default=32)
    b.add_argument("--tile_w", type=int, default=32)
    b.add_argument("--in_channels", type=int, default=16)
    b.add_argument("--out_channels", type=int, default=256)
    b.add_argument("--global_h", type=int, default=625)
    b.add_argument("--global_w", type=int, default=625)
    b.add_argument("--overlap", type=int, default=8)
    b.add_argument("--keep_frac", type=float, default=0.25, help="Spectral export fraction. 1.0 = lossless strict replay; 0.25 = lossy demo/compression.")

    b.set_defaults(fn=cmd_build_global)

    v = sub.add_parser("validate-recon", help="Validate replay reconstruction accuracy")
    v.add_argument("--out_dir", default=DEFAULT_OUT)
    v.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    v.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    v.add_argument("--fig_path", default="benchmarks/results/figures/replay_error_heatmap.png")
    v.add_argument("--replay_only", action="store_true", help="Delete global, rebuild from coeffs, validate vs backup")
    v.set_defaults(fn=cmd_validate_recon)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
