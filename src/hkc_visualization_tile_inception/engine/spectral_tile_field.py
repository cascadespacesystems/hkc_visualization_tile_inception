# FILE: src/hkc_visualization_tile/engine/spectral_tile_field.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .overlap_add import OverlapAddPlan, overlap_add_accumulate, overlap_add_finalize
from ..io.manifest import ManifestV1, TileEntryV1
from ..io.tile_file_io import save_pt, save_json, load_pt, load_json


@dataclass(frozen=True)
class SpectralTileFieldConfig:
    tile_h: int = 32
    tile_w: int = 32
    in_channels: int = 16
    out_channels: int = 256
    global_h: int = 625
    global_w: int = 625
    overlap: int = 8
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == "float32" else torch.float16


def _device(cfg: SpectralTileFieldConfig) -> torch.device:
    ds = cfg.device.lower()
    if ds == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if ds == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_synthetic_tiles(cfg: SpectralTileFieldConfig) -> List[torch.Tensor]:
    """
    Generates a list of synthetic tiles [th, tw, cin].
    Tile count is determined by the overlap-add plan to cover the global area.
    """
    dev = _device(cfg)
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)

    plan = OverlapAddPlan(cfg.global_h, cfg.global_w, cfg.tile_h, cfg.tile_w, cfg.overlap)
    tops = list(range(0, cfg.global_h, plan.stride_h))
    lefts = list(range(0, cfg.global_w, plan.stride_w))

    tiles: List[torch.Tensor] = []
    for _t in tops:
        for _l in lefts:
            # Smooth-ish noise: random + small blur in spatial domain
            base = torch.randn((cfg.tile_h, cfg.tile_w, cfg.in_channels), generator=g, dtype=torch.float32)
            base = base.to(dev, dtype=cfg.torch_dtype)
            tiles.append(base)
    return tiles


def lift_channels(tile: torch.Tensor, out_channels: int, seed: int = 0) -> torch.Tensor:
    """
    Deterministic channel lift cin -> cout using a fixed random projection.
    This is public-safe and demonstrates the dimensional lift without proprietary kernels.
    """
    th, tw, cin = tile.shape
    dev = tile.device
    dtype = tile.dtype

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn((cin, out_channels), generator=g, dtype=torch.float32).to(dev, dtype=dtype) / (cin ** 0.5)
    # [th, tw, cin] @ [cin, cout] => [th, tw, cout]
    return tile.reshape(-1, cin).matmul(W).reshape(th, tw, out_channels)


def build_global_from_tiles(cfg: SpectralTileFieldConfig, tiles: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]]]:
    """
    Returns:
      global_X: [H, W, C]
      weight_acc: [H, W, 1]
      placements: list of (top,left) corresponding to tiles list order
    """
    dev = _device(cfg)
    plan = OverlapAddPlan(cfg.global_h, cfg.global_w, cfg.tile_h, cfg.tile_w, cfg.overlap)

    acc = torch.zeros((cfg.global_h, cfg.global_w, cfg.out_channels), device=dev, dtype=cfg.torch_dtype)
    wacc = torch.zeros((cfg.global_h, cfg.global_w, 1), device=dev, dtype=cfg.torch_dtype)

    tops = list(range(0, cfg.global_h, plan.stride_h))
    lefts = list(range(0, cfg.global_w, plan.stride_w))

    placements: List[Tuple[int,int]] = []
    idx = 0
    for top in tops:
        for left in lefts:
            tile_in = tiles[idx].to(dev)
            tile_out = lift_channels(tile_in, cfg.out_channels, seed=cfg.seed)
            overlap_add_accumulate(acc, wacc, tile_out, top=top, left=left)
            placements.append((top, left))
            idx += 1

    X = overlap_add_finalize(acc, wacc)
    return X, wacc, placements


def shell_compress_rfft(tile: torch.Tensor, keep_frac: float = 0.25) -> dict:
    """
    Export rFFT2 coefficients. Two modes:
      - Lossless: keep_frac >= 1.0  -> store full spectrum
      - Lossy:    keep_frac <  1.0  -> store cropped low-frequency block
    """
    th, tw, C = tile.shape
    F = torch.fft.rfft2(tile, dim=(0, 1))  # complex

    # Lossless mode
    if keep_frac >= 1.0:
        return {
            "schema": "tile_coeff_v1",
            "tile_shape": [th, tw, C],
            "keep_frac": 1.0,
            "kh": int(th),
            "kw": int(tw // 2 + 1),
            "coeff": F.cpu(),  # full spectrum, exact replay
        }

    # Lossy mode (cropped spectrum)
    kh = max(1, int(th * keep_frac))
    kw = max(1, int((tw // 2 + 1) * keep_frac))
    Fk = F[:kh, :kw, :].contiguous()

    return {
        "schema": "tile_coeff_v1",
        "tile_shape": [th, tw, C],
        "keep_frac": float(keep_frac),
        "kh": int(kh),
        "kw": int(kw),
        "coeff": Fk.cpu(),
    }


def shell_decompress_rfft(payload: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    th, tw, C = payload["tile_shape"]
    kh = payload["kh"]
    kw = payload["kw"]
    coeff = payload["coeff"].to(device=device)

    # create full spectrum, place low-freq block
    F = torch.zeros((th, tw // 2 + 1, C), device=device, dtype=torch.complex64)
    F[:kh, :kw, :] = coeff.to(torch.complex64)

    # invert
    x = torch.fft.irfft2(F, s=(th, tw), dim=(0, 1)).to(dtype=dtype)
    return x


def export_tile_coeffs_and_manifest(
    cfg: SpectralTileFieldConfig,
    X: torch.Tensor,
    placements: List[Tuple[int,int]],
    out_dir: str,
    keep_frac: float = 0.25,
) -> ManifestV1:
    """
    Exports per-tile coefficients from the built global tensor X.
    This is intentionally public: coefficients + manifest define replayability.
    """
    os.makedirs(out_dir, exist_ok=True)
    coeff_dir = os.path.join(out_dir, "tile_coeffs")
    os.makedirs(coeff_dir, exist_ok=True)

    entries: List[TileEntryV1] = []
    tile_id = 0
    for (top, left) in placements:
        th, tw = cfg.tile_h, cfg.tile_w
        tile = X[top:top+th, left:left+tw, :].contiguous()
        payload = shell_compress_rfft(tile, keep_frac=keep_frac)
        coeff_path = os.path.join("tile_coeffs", f"tile_{tile_id:06d}.pt")
        save_pt(os.path.join(out_dir, coeff_path), payload)

        entries.append(TileEntryV1(
            tile_id=tile_id,
            top=int(top),
            left=int(left),
            height=int(th),
            width=int(tw),
            coeff_path=coeff_path,
        ))
        tile_id += 1

    manifest = ManifestV1(
        schema="hkc_manifest",
        version=1,
        global_shape=[cfg.global_h, cfg.global_w, cfg.out_channels],
        tile_shape=[cfg.tile_h, cfg.tile_w, cfg.in_channels],
        lifted_channels=cfg.out_channels,
        overlap=cfg.overlap,
        tiles_total=len(entries),
        entries=entries,
    )
    save_json(os.path.join(out_dir, "manifest_tiles.json"), manifest.to_dict())
    return manifest


def rebuild_global_from_coeffs(manifest_path: str, out_dir: str, device: str = "cpu", dtype: str = "float32") -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Rebuilds global tensor using coeff tiles + manifest only.
    Returns (X_rebuilt, wacc, summary_dict)
    """
    md = load_json(manifest_path)
    manifest = ManifestV1.from_dict(md)

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available()
                       else "mps" if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                       else "cpu")
    tdtype = torch.float32 if dtype == "float32" else torch.float16

    H, W, C = manifest.global_shape
    acc = torch.zeros((H, W, C), device=dev, dtype=tdtype)
    wacc = torch.zeros((H, W, 1), device=dev, dtype=tdtype)

    used = 0
    missing = 0
    for e in manifest.entries:
        p = os.path.join(out_dir, e.coeff_path)
        if not os.path.exists(p):
            missing += 1
            continue
        payload = load_pt(p, map_location="cpu")
        tile = shell_decompress_rfft(payload, device=dev, dtype=tdtype)  # [th, tw, C]
        overlap_add_accumulate(acc, wacc, tile, top=e.top, left=e.left)
        used += 1

    X = overlap_add_finalize(acc, wacc)

    summary = {
        "tiles_total_in_manifest": int(manifest.tiles_total),
        "tiles_used": int(used),
        "tiles_missing_coeff_path": int(missing),
        "global_shape": [int(H), int(W), int(C)],
    }
    return X, wacc, summary
