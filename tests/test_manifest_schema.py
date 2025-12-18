# FILE: tests/test_manifest_schema.py
from hkc_visualization_tile_inception.io.manifest import ManifestV1, TileEntryV1


def test_manifest_roundtrip():
    entries = [TileEntryV1(tile_id=0, top=0, left=0, height=32, width=32, coeff_path="tile_coeffs/tile_000000.pt")]
    m = ManifestV1(
        schema="hkc_manifest",
        version=1,
        global_shape=[10, 10, 8],
        tile_shape=[32, 32, 16],
        lifted_channels=256,
        overlap=8,
        tiles_total=1,
        entries=entries,
    )
    d = m.to_dict()
    m2 = ManifestV1.from_dict(d)
    assert m2.schema == m.schema
    assert m2.version == 1
    assert m2.entries[0].coeff_path.endswith(".pt")
