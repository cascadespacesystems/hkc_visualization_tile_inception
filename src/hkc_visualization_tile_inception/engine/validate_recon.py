# FILE: src/hkc_visualization_tile/engine/validate_recon.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from .spectral_tile_field import rebuild_global_from_coeffs
from ..io.tile_file_io import load_pt, save_pt
from ..viz.error_heatmap import save_error_heatmap


@dataclass(frozen=True)
class ReconSummary:
    mae: float
    rmse: float
    maxe: float
    coverage_frac: float
    tiles_used: int
    tiles_total: int
    tiles_missing: int
    paths: Dict[str, str]


def validate_reconstruction(
    manifest_path: str,
    coeff_root: str,
    saved_global_path: str,
    saved_weight_path: str,
    device: str = "cpu",
    dtype: str = "float32",
    out_fig_path: str | None = None,
) -> ReconSummary:
    """
    Rebuild from coeff tiles + manifest, compare against saved global.
    """
    X_ref = load_pt(saved_global_path, map_location="cpu")
    W_ref = load_pt(saved_weight_path, map_location="cpu")

    X_rec, W_rec, meta = rebuild_global_from_coeffs(
        manifest_path=manifest_path,
        out_dir=coeff_root,
        device=device,
        dtype=dtype,
    )

    X_ref = X_ref.to(dtype=torch.float32, device="cpu")
    X_rec_cpu = X_rec.detach().to(dtype=torch.float32, device="cpu")

    # coverage from reconstructed weight map (nonzero weights)
    W_rec_cpu = W_rec.detach().to(dtype=torch.float32, device="cpu")
    covered = (W_rec_cpu[..., 0] > 0).float()
    coverage_frac = float(covered.mean().item())

    diff = (X_rec_cpu - X_ref)
    mae = float(diff.abs().mean().item())
    rmse = float(torch.sqrt((diff * diff).mean()).item())
    maxe = float(diff.abs().max().item())

    if out_fig_path:
        # heatmap: mean abs error across channels
        err2d = diff.abs().mean(dim=-1)  # [H,W]
        save_error_heatmap(err2d, out_fig_path)

    return ReconSummary(
        mae=mae,
        rmse=rmse,
        maxe=maxe,
        coverage_frac=coverage_frac,
        tiles_used=int(meta["tiles_used"]),
        tiles_total=int(meta["tiles_total_in_manifest"]),
        tiles_missing=int(meta["tiles_missing_coeff_path"]),
        paths={
            "manifest": manifest_path,
            "global": saved_global_path,
            "weight": saved_weight_path,
        },
    )
