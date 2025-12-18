import os
import tempfile
import torch

from hkc_visualization_tile_inception.engine.spectral_tile_field import (
    SpectralTileFieldConfig,
    make_synthetic_tiles,
    build_global_from_tiles,
    export_tile_coeffs_and_manifest,
    rebuild_global_from_coeffs,
)

def _rebuild_err(cfg, X, placements, keep_frac):
    with tempfile.TemporaryDirectory() as d:
        export_tile_coeffs_and_manifest(cfg, X, placements, out_dir=d, keep_frac=keep_frac)
        X2, _, _ = rebuild_global_from_coeffs(
            manifest_path=os.path.join(d, "manifest_tiles.json"),
            out_dir=d,
            device="cpu",
            dtype="float32",
        )
        return float((X2.cpu().float() - X.cpu().float()).abs().mean().item())

def test_lossy_error_decreases_with_more_spectrum():
    cfg = SpectralTileFieldConfig(global_h=96, global_w=96, out_channels=32, tile_h=32, tile_w=32, overlap=8, device="cpu")
    tiles = make_synthetic_tiles(cfg)
    X, _, placements = build_global_from_tiles(cfg, tiles)

    e25 = _rebuild_err(cfg, X, placements, keep_frac=0.25)
    e75 = _rebuild_err(cfg, X, placements, keep_frac=0.75)

    assert e75 < e25
    assert torch.isfinite(torch.tensor([e25, e75])).all()
