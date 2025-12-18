# FILE: src/hkc_visualization_tile/io/tile_file_io.py
from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pt(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(obj, path)


def load_pt(path: str, map_location: str | None = None) -> Any:
    return torch.load(path, map_location=map_location)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
