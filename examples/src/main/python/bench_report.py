"""Combine the two benchmark reports into one table, and check the two programs agree.

Reads `deepwit_bench.json` / `python_bench.json` and the two loss traces written by
`AutoEncoderBench` and `autoencoder_bench.py`, and prints the comparison that goes in
the paper: per-step time for each of the four configurations, and the numerical
agreement between the DeepWit and JAX training trajectories from a shared starting point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO_ROOT / "examples" / "out" / "bench"


def read_trace(path: Path) -> list[float]:
    with open(path) as f:
        return [float(row["loss"]) for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with open(args.out / "deepwit_bench.json") as f:
        scala = json.load(f)
    with open(args.out / "python_bench.json") as f:
        python = json.load(f)

    print(f"Device:      {scala['device']}  /  {python['device']}")
    print(f"JAX:         {scala['jax_version']}  /  {python['jax_version']}")
    print(f"Batch size:  {scala['batch_size']}, latent {scala['latent_dim']}, lr {scala['learning_rate']}")
    print(f"Warm-up:     {scala['warmup_steps']} steps discarded per configuration")

    print("\n-- Numerical equivalence --")
    checksum_delta = abs(scala["batch_checksum"] - python["batch_checksum"])
    print(f"Batch checksum: {scala['batch_checksum']:.6f} vs {python['batch_checksum']:.6f}  (|Δ| = {checksum_delta:.6g})")

    deepwit_trace = read_trace(args.out / "scala_loss_trace.csv")
    jax_trace = read_trace(args.out / "python_loss_trace.csv")
    n = min(len(deepwit_trace), len(jax_trace))
    abs_errors = [abs(a - b) for a, b in zip(deepwit_trace[:n], jax_trace[:n])]
    rel_errors = [e / abs(a) for e, a in zip(abs_errors, deepwit_trace[:n]) if a != 0.0]
    print(f"Loss trace over {n} steps, from identical parameters:")
    print(f"  step 1:  deepwit {deepwit_trace[0]:.6f}   jax {jax_trace[0]:.6f}")
    print(f"  step {n}: deepwit {deepwit_trace[n - 1]:.6f}   jax {jax_trace[n - 1]:.6f}")
    print(f"  max |Δ| = {max(abs_errors):.6g},  max relative Δ = {max(rel_errors):.6g}")

    print("\n-- Per-step time --")
    header = (
        f"{'configuration':<16} {'mean ms':>9} {'std':>8} {'median':>9} "
        f"{'min':>9} {'max':>9} {'steps/s':>9} {'images/s':>10} {'1st call ms':>12} {'n':>8}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for report in (scala, python):
        for m in report["measurements"]:
            rows.append((report, m))
    for report, m in rows:
        print(
            f"{m['label']:<16} {m['mean_step_ms']:9.3f} {m['std_step_ms']:8.3f} {m['median_step_ms']:9.3f} "
            f"{m['min_step_ms']:9.3f} {m['max_step_ms']:9.3f} {m['steps_per_second']:9.1f} "
            f"{m['steps_per_second'] * report['batch_size']:10.0f} {m['first_call_ms']:12.1f} "
            f"{m['reps']}x{m['steps_per_rep']:<6}"
        )

    by_label = {m["label"]: m for _, m in rows}

    print("\n-- Ratios (DeepWit / JAX, per-step time) --")
    for a, b in (("deepwit-jit", "jax-jit"), ("deepwit-eager", "jax-eager")):
        if a in by_label and b in by_label:
            print(f"  {a} / {b}: {by_label[a]['mean_step_ms'] / by_label[b]['mean_step_ms']:.2f}x")
    for impl, (jit, eager) in (("deepwit", ("deepwit-jit", "deepwit-eager")), ("jax", ("jax-jit", "jax-eager"))):
        if jit in by_label and eager in by_label:
            print(f"  {impl}: JIT is {by_label[eager]['mean_step_ms'] / by_label[jit]['mean_step_ms']:.1f}x faster than eager")

    with open(args.out / "comparison.json", "w") as f:
        json.dump(
            {
                "device": scala["device"],
                "jax_version": scala["jax_version"],
                "equivalence": {
                    "batch_checksum_delta": checksum_delta,
                    "loss_trace_steps": n,
                    "max_abs_loss_delta": max(abs_errors),
                    "max_rel_loss_delta": max(rel_errors),
                    "deepwit_loss_trace": deepwit_trace[:n],
                    "jax_loss_trace": jax_trace[:n],
                },
                "measurements": [m for _, m in rows],
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out / 'comparison.json'}")


if __name__ == "__main__":
    main()
