# FILE: benchmarks/README.md
# Benchmarks & Validation

This directory contains reproducible evidence of:

- optimizer behavior (steps-to-hit)
- wall-clock runtime (time-to-hit)
- deterministic replay correctness (error heatmap)

Figures are written to:

- `benchmarks/results/figures/steps_to_hit.png`
- `benchmarks/results/figures/wall_clock.png`
- `benchmarks/results/figures/replay_error_heatmap.png`

Generate benchmark figures:

```bash
bash tools/run_benchmark_figures.sh
