# FILE: src/hkc_visualization_tile/io/manifest.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class TileEntryV1:
    tile_id: int
    top: int
    left: int
    height: int
    width: int
    coeff_path: str


@dataclass(frozen=True)
class ManifestV1:
    schema: str
    version: int
    global_shape: List[int]          # [H, W, C]
    tile_shape: List[int]            # [th, tw, cin]
    lifted_channels: int             # cout
    overlap: int
    tiles_total: int
    entries: List[TileEntryV1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "global_shape": self.global_shape,
            "tile_shape": self.tile_shape,
            "lifted_channels": self.lifted_channels,
            "overlap": self.overlap,
            "tiles_total": self.tiles_total,
            "entries": [asdict(e) for e in self.entries],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ManifestV1":
        entries = [TileEntryV1(**e) for e in d["entries"]]
        return ManifestV1(
            schema=str(d["schema"]),
            version=int(d["version"]),
            global_shape=list(d["global_shape"]),
            tile_shape=list(d["tile_shape"]),
            lifted_channels=int(d["lifted_channels"]),
            overlap=int(d["overlap"]),
            tiles_total=int(d["tiles_total"]),
            entries=entries,
        )
