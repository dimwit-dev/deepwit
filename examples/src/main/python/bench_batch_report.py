"""Join the two batch sweeps: does more work per call hide DeepWit's fixed bridge cost?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO_ROOT / "examples" / "out" / "bench"


def by_kind(config: dict, kind: str) -> dict | None:
    for m in config["measurements"]:
        if f"-{kind}-" in m["label"]:
            return m
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with open(args.out / "deepwit_batch_bench.json") as f:
        scala = json.load(f)
    with open(args.out / "python_batch_bench.json") as f:
        python = json.load(f)

    print(f"Device:      {scala['device']}  /  {python['device']}")
    print(f"JAX:         {scala['jax_version']}  /  {python['jax_version']}")
    print(f"Model:       {scala.get('model', 'unknown')}, {scala['leaves']} tensors in the train state")
    print(f"Bridge:      {scala['round_trip_ms']:.3f} ms round trip, the same at every batch size")
    print(f"Warm-up:     {scala['warmup_steps']} steps discarded per configuration")

    python_by_batch = {c["batch_size"]: c for c in python["configs"]}

    for kind in ("jit", "eager"):
        print(f"\n-- {kind.upper()} --")
        header = (
            f"{'batch':>7} {'deepwit ms':>12} {'jax ms':>10} {'gap ms':>9} {'ratio':>7} "
            f"{'deepwit img/s':>15} {'jax img/s':>12} {'throughput kept':>16}"
        )
        print(header)
        print("-" * len(header))
        for c in scala["configs"]:
            batch = c["batch_size"]
            s_m = by_kind(c, kind)
            p_c = python_by_batch.get(batch)
            p_m = by_kind(p_c, kind) if p_c else None
            if s_m is None or p_m is None:
                continue
            s_img = s_m["steps_per_second"] * batch
            p_img = p_m["steps_per_second"] * batch
            print(
                f"{batch:>7} {s_m['mean_step_ms']:>12.3f} {p_m['mean_step_ms']:>10.3f} "
                f"{s_m['mean_step_ms'] - p_m['mean_step_ms']:>9.3f} "
                f"{s_m['mean_step_ms'] / p_m['mean_step_ms']:>7.2f} "
                f"{s_img:>15,.0f} {p_img:>12,.0f} {100.0 * s_img / p_img:>15.1f}%"
            )

    combined = {
        "device": scala["device"],
        "model": "autoencoder-6layer",
        "leaves": scala["leaves"],
        "round_trip_ms": scala["round_trip_ms"],
        "configs": [
            {
                "batch_size": c["batch_size"],
                "deepwit": c["measurements"],
                "jax": python_by_batch.get(c["batch_size"], {}).get("measurements", []),
            }
            for c in scala["configs"]
        ],
    }
    with open(args.out / "batch_comparison.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {args.out / 'batch_comparison.json'}")


if __name__ == "__main__":
    main()
