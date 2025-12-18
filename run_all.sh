#!/usr/bin/env bash
set -euo pipefail

# run_all.sh — one-shot, sequential end-to-end run for hkc_visualization_tile_inception
# Usage:
#   bash run_all.sh
#   DEVICE=cuda bash run_all.sh
#   DEVICE=mps  bash run_all.sh
#
# Notes:
# - Picks DEVICE env var (default: cpu). Falls back automatically if cuda/mps unavailable.
# - Generates artifacts + replay validation + benchmark JSON + figures + tests.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PKG="hkc_visualization_tile_inception"
OUT_DIR="artifacts/harmonic_export"
BENCH_DIR="benchmarks/results"
FIG_DIR="benchmarks/results/figures"
DEVICE="${DEVICE:-cpu}"

echo "[run_all] repo:   $ROOT_DIR"
echo "[run_all] device: $DEVICE"
echo

# -----------------------------
# 0) Venv + install
# -----------------------------
if [[ ! -d ".venv" ]]; then
  echo "[0] creating venv..."
  python -m venv .venv
fi

echo "[0] activating venv..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[0] upgrading pip..."
pip3 install -U pip >/dev/null

echo "[0] editable install + dev deps..."
pip3 install -e ".[dev]" >/dev/null

# -----------------------------
# 1) Sanity: ensure no stale imports
# -----------------------------
echo
echo "[1] checking for stale imports (hkc_visualization_tile)..."
STALE="$(grep -R "\bhkc_visualization_tile\b" -n src benchmarks tools tests \
  --exclude-dir='*.egg-info' \
  --exclude='PKG-INFO' \
  --exclude='SOURCES.txt' \
  --exclude='entry_points.txt' \
  --exclude='requires.txt' \
  --exclude='top_level.txt' \
  | grep -v "^src/.*:1:# FILE:" \
  || true)"

if [[ -n "$STALE" ]]; then
  echo "[error] Found stale imports referencing old package name:"
  echo "$STALE"
  exit 2
fi
echo "[1] OK"

# -----------------------------
# 2) Clean output dirs (keep it deterministic)
# -----------------------------
echo
echo "[2] preparing output dirs..."
mkdir -p "$OUT_DIR" "$BENCH_DIR" "$FIG_DIR"

# Optional: remove prior backups to keep the demo clean
rm -f "$OUT_DIR"/_backup_global_*.pt "$OUT_DIR"/_backup_global_weight_acc.pt || true

# -----------------------------
# 3) Build global + export coeffs + manifest
# -----------------------------
echo
echo "[3] build-global..."
python3 -m "${PKG}.cli.hkc_tile" build-global --out_dir "$OUT_DIR" --device "$DEVICE"

# -----------------------------
# 4) Validate replay-only (delete global, rebuild from coeffs+manifest, compare to backup)
# -----------------------------
echo
echo "[4] validate-recon --replay_only..."
python3 -m "${PKG}.cli.hkc_tile" validate-recon --out_dir "$OUT_DIR" --device "$DEVICE" --replay_only

# -----------------------------
# 5) Run baseline benchmark (writes JSON)
# -----------------------------
echo
echo "[5] baseline benchmark..."
python3 benchmarks/bench_grid_adam.py --device cpu --out "${BENCH_DIR}/bench_cpu.json"

# -----------------------------
# 6) Plot figures (steps_to_hit.png + wall_clock.png)
# -----------------------------
echo
echo "[6] plot benchmark figures..."
bash tools/run_benchmark_figures.sh

# -----------------------------
# 7) Tests
# -----------------------------
echo
echo "[7] pytest..."
pytest

echo
echo "[run_all] COMPLETE ✅"
echo "Artifacts:"
echo "  - ${OUT_DIR}/manifest_tiles.json"
echo "  - ${OUT_DIR}/tile_coeffs/"
echo "  - ${OUT_DIR}/global_*.pt"
echo "Figures:"
echo "  - ${FIG_DIR}/replay_error_heatmap.png"
echo "  - ${FIG_DIR}/steps_to_hit.png"
echo "  - ${FIG_DIR}/wall_clock.png"
echo "Bench JSON:"
echo "  - ${BENCH_DIR}/bench_cpu.json"
