# FILE: tests/test_tile_io.py
import os
import tempfile
import torch
from hkc_visualization_tile_inception.io.tile_file_io import save_pt, load_pt, save_json, load_json


def test_pt_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.pt")
        x = {"a": torch.randn(3, 4)}
        save_pt(p, x)
        y = load_pt(p, map_location="cpu")
        assert "a" in y
        assert y["a"].shape == (3, 4)


def test_json_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        obj = {"schema": "x", "v": 1}
        save_json(p, obj)
        obj2 = load_json(p)
        assert obj2["v"] == 1
