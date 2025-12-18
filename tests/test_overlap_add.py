# FILE: tests/test_overlap_add.py
import torch
from hkc_visualization_tile_inception.engine.overlap_add import overlap_add_accumulate, overlap_add_finalize


def test_overlap_add_shapes():
    H, W, C = 64, 64, 8
    acc = torch.zeros((H, W, C))
    wacc = torch.zeros((H, W, 1))

    tile = torch.ones((16, 16, C))
    overlap_add_accumulate(acc, wacc, tile, top=0, left=0)
    overlap_add_accumulate(acc, wacc, tile, top=8, left=8)

    out = overlap_add_finalize(acc, wacc)
    assert out.shape == (H, W, C)
    assert torch.isfinite(out).all()
