
# hkc_visualization_tile_inception
=======

**Spectral Tile Field Engine** for deterministic construction, replay, and
validation of large tensor fields on GPUs.

This repository is the **public, Inception-safe** productized form of the
original experimental work: it demonstrates replayable spectral artifacts,
manifest-driven reconstruction, and numerical validation—without exposing
proprietary frameworks.

---

## What it does

Default pipeline:

- ingest tiles (default: synthetic **32×32×16**)
- lift channels (**16 → 256**)
- overlap-add assemble global field (default: **625×625×256**)
- export **shell-compressed rFFT coefficients per tile**
- write a **manifest** mapping `tile_id → (top,left) → coeff_path`
- validate replayability: coeff tiles → rebuild global → compare to saved global

---

## Quickstart (dev install)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

Flagship demo: deterministic replay
1) Build global + export coeffs + manifest
python -m hkc_visualization_tile.cli.hkc_tile build-global


Writes:

artifacts/harmonic_export/global_625x625x256.pt

artifacts/harmonic_export/global_weight_acc.pt

artifacts/harmonic_export/manifest_tiles.json

artifacts/harmonic_export/tile_coeffs/tile_XXXXXX.pt

2) Validate replayability (full proof)
python -m hkc_visualization_tile.cli.hkc_tile validate-recon


This rebuilds the global tensor from coeff tiles + manifest only and reports:

MAE / RMSE / max error

coverage fraction

writes benchmarks/results/figures/replay_error_heatmap.png

Benchmarks & validation panel

steps_to_hit.png

wall_clock.png

replay_error_heatmap.png

Generate figures:

bash tools/run_benchmark_figures.sh

## Two export modes (professional + reviewer-friendly)

This engine supports **two spectral export modes**:

### Lossless mode (keep_frac = 1.0)
Use this for:
- unit tests
- strict replay proofs
- CI reliability

Command:

python3 -m hkc_visualization_tile_inception.cli.hkc_tile build-global --keep_frac 1.0
python3 -m hkc_visualization_tile_inception.cli.hkc_tile validate-recon --replay_only

```
>>>>>>> 970268e (Inception-Public)
