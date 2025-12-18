# FILE: tests/test_replay_determinism.py
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
from hkc_visualization_tile_inception.io.tile_file_io import save_pt


def test_replay_rebuild_close():
    cfg = SpectralTileFieldConfig(global_h=96, global_w=96, out_channels=32, tile_h=32, tile_w=32, overlap=8, device="cpu")
    tiles = make_synthetic_tiles(cfg)
    X, wacc, placements = build_global_from_tiles(cfg, tiles)

    with tempfile.TemporaryDirectory() as d:
        # save global and weight
        gpath = os.path.join(d, "global.pt")
        wpath = os.path.join(d, "wacc.pt")
        save_pt(gpath, X.detach().cpu())
        save_pt(wpath, wacc.detach().cpu())

        m = export_tile_coeffs_and_manifest(cfg, X, placements, out_dir=d, keep_frac=1.0)
        X2, w2, meta = rebuild_global_from_coeffs(
            manifest_path=os.path.join(d, "manifest_tiles.json"),
            out_dir=d,
            device="cpu",
            dtype="float32",
        )

        # compare
        diff = (X2.detach().cpu().float() - X.detach().cpu().float()).abs().mean().item()
        assert diff < 1e-5  # compression introduces small error; keep threshold reasonable
