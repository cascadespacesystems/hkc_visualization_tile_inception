# FILE: tools/run_benchmark_figures.sh
#!/usr/bin/env bash
set -euo pipefail

python3 tools/plot_benchmark_figures.py \
  --results_dir benchmarks/results \
  --fig_dir benchmarks/results/figures
