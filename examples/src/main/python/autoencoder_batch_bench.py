"""The six-layer autoencoder benchmark swept over batch size, in plain JAX.

The counterpart to `benchAutoEncoderBatch` in `AutoEncoderBench.scala`. Unlike the depth sweep,
batch size adds work without adding parameters, so this is the axis on which DeepWit's fixed
per-call bridge cost gets something to hide behind.

Model, loss, optimizer and measurement protocol are all imported from `autoencoder_bench`, so
only the batch differs between points.

Run, from the repository root:
    uv run --no-sync python examples/src/main/python/autoencoder_batch_bench.py --batches 64,256,1024,4096,16384
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from autoencoder_bench import (
    EXAMPLES_ROOT,
    TrainState,
    adam_init,
    gradient_step,
    init_params,
    load_exported_params,
    load_train_images,
    measure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=str, default="64,256,1024,4096,16384")
    parser.add_argument("--steps", type=int, default=100, help="JIT steps per timed repetition")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eager-steps", type=int, default=5)
    parser.add_argument("--eager-reps", type=int, default=8)
    parser.add_argument("--mode", choices=("both", "jit", "eager"), default="both")
    parser.add_argument("--out", type=Path, default=EXAMPLES_ROOT / "out" / "bench")
    parser.add_argument("--data", type=Path, default=EXAMPLES_ROOT / "data")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(b) for b in args.batches.split(",")]

    device = jax.devices()[0]
    print(f"Device: {device}")
    print(f"JAX:    {jax.__version__}")

    images = load_train_images(args.data)

    exported = args.out / "init_params.pkl"
    if exported.exists():
        params = load_exported_params(exported)
        print(f"Parameters: loaded from {exported.name}")
    else:
        params = init_params(jax.random.key(42))
        print("Parameters: freshly initialized (no DeepWit export found)")

    leaves = len(jax.tree.leaves(params)) * 3 + 3
    print(f"Train state: {leaves} tensors")

    def initial() -> TrainState:
        fresh = jax.tree.map(lambda x: x.copy(), params)
        return TrainState(fresh, adam_init(fresh), jnp.float32(-1.0))

    configs = []
    for batch_size in batch_sizes:
        batch = jax.block_until_ready(images[:batch_size])
        print(f"\n== batch {batch_size} ==")

        measurements = []
        if args.mode in ("both", "jit"):
            # A fresh jit per batch size, so its first call is a genuinely cold compile.
            jit_step = jax.jit(gradient_step, donate_argnums=(1,))
            measurements.append(
                measure(f"jax-jit-b{batch_size}", jit_step, batch, initial, args.warmup, args.reps, args.steps)
            )
        if args.mode in ("both", "eager"):
            measurements.append(
                measure(
                    f"jax-eager-b{batch_size}",
                    gradient_step,
                    batch,
                    initial,
                    args.warmup,
                    args.eager_reps,
                    args.eager_steps,
                )
            )
        for m in measurements:
            print(
                f"{m['label']:<28} {m['mean_step_ms']:9.3f} ± {m['std_step_ms']:<8.3f} "
                f"{m['median_step_ms']:9.3f} {m['min_step_ms']:9.3f} {m['max_step_ms']:9.3f} "
                f"{m['steps_per_second']:9.1f} {m['first_call_ms']:10.1f}"
            )

        configs.append({"batch_size": batch_size, "measurements": measurements})

    report = {
        "implementation": "jax-python",
        "model": "autoencoder-6layer",
        "device": str(device),
        "jax_version": jax.__version__,
        "warmup_steps": args.warmup,
        "leaves": leaves,
        "configs": configs,
    }
    with open(args.out / "python_batch_bench.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out / 'python_batch_bench.json'}")


if __name__ == "__main__":
    main()
