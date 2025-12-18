# FILE: tools/plot_benchmark_figures.py
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_runs(results_dir: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.getsize(p) == 0:
            continue
        try:
            obj = read_json(p)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("runs"), list):
            for r in obj["runs"]:
                if isinstance(r, dict):
                    runs.append(r)
    return runs


def median(vals: List[float]) -> float:
    vs = sorted(vals)
    n = len(vs)
    if n == 0:
        return float("nan")
    m = n // 2
    return vs[m] if n % 2 == 1 else 0.5 * (vs[m - 1] + vs[m])


def plot_steps_to_hit(runs: List[Dict[str, Any]], out_png: str) -> None:
    # label: mode [device]
    buckets: Dict[str, List[float]] = {}
    nohit: Dict[str, int] = {}
    total: Dict[str, int] = {}

    for r in runs:
        mode = str(r.get("mode", "unknown"))
        dev = str(r.get("device", "unknown")).replace("torch.", "")
        label = f"{mode} [{dev}]"
        total[label] = total.get(label, 0) + 1

        hit = r.get("first_hit_step", None)
        steps_ran = int(r.get("steps_ran", r.get("steps", 0) or 0))
        if hit is None:
            v = float(max(steps_ran, 0))
            nohit[label] = nohit.get(label, 0) + 1
        else:
            v = float(hit)
        buckets.setdefault(label, []).append(v)

    labels = sorted(buckets.keys())
    values = [median(buckets[k]) for k in labels]
    counts = [len(buckets[k]) for k in labels]
    nohit_rate = [nohit.get(k, 0) / max(total.get(k, 1), 1) for k in labels]

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Steps to Hit Target (Lower is Better) — NO-HIT capped at step budget")
    plt.ylabel("Steps (median)")
    plt.xlabel("Optimizer [device]")

    x = list(range(len(labels)))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=35, ha="right")

    for i, (v, n, r) in enumerate(zip(values, counts, nohit_rate)):
        tag = "HIT" if r < 0.5 else ("MIX" if r < 1.0 else "NO-HIT")
        plt.text(i, v, f"{tag}\n n={n}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_wall_clock(runs: List[Dict[str, Any]], out_png: str) -> None:
    buckets: Dict[str, List[float]] = {}
    for r in runs:
        t = r.get("time_s", None)
        if t is None:
            continue
        mode = str(r.get("mode", "unknown"))
        dev = str(r.get("device", "unknown")).replace("torch.", "")
        label = f"{mode} [{dev}]"
        buckets.setdefault(label, []).append(float(t))

    labels = sorted(buckets.keys())
    values = [median(buckets[k]) for k in labels]
    counts = [len(buckets[k]) for k in labels]

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Wall Clock Time to Target (Lower is Better)")
    plt.ylabel("Seconds (median)")
    plt.xlabel("Optimizer [device]")

    x = list(range(len(labels)))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=35, ha="right")

    for i, (v, n) in enumerate(zip(values, counts)):
        plt.text(i, v, f"n={n}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate steps_to_hit.png and wall_clock.png from bench_v1 JSON.")
    ap.add_argument("--results_dir", default="benchmarks/results")
    ap.add_argument("--fig_dir", default="benchmarks/results/figures")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit("[error] no valid bench_v1 JSON runs found in benchmarks/results/")

    steps_png = os.path.join(args.fig_dir, "steps_to_hit.png")
    wall_png = os.path.join(args.fig_dir, "wall_clock.png")
    os.makedirs(args.fig_dir, exist_ok=True)

    plot_steps_to_hit(runs, steps_png)
    plot_wall_clock(runs, wall_png)
    print(f"[ok] wrote {steps_png}")
    print(f"[ok] wrote {wall_png}")


if __name__ == "__main__":
    main()
