"""Steady-state cost of one autoencoder training step in plain JAX, JIT-compiled and eager.

The reference this benchmarks against is
`examples/src/main/scala/deepwit/examples/autoencoder/AutoEncoderBench.scala`, which measures
the DeepWit program of the same shape under the same protocol. Equivalence is structural,
not just numerical: the model is written as a per-sample function under `jax.vmap`, the
reconstruction loss as a per-pixel function under a second `jax.vmap`, and the optimizer as
DeepWit's `Adam` transcribed term for term — because that is what DeepWit's `vmap`, `zipvmap`
and `Adam` lower to. A batched matmul formulation would measure a different program.

Run, from the repository root:
    uv run --no-sync python examples/src/main/python/autoencoder_bench.py --steps 100 --reps 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import struct
import time
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLES_ROOT = REPO_ROOT / "examples"

# -- Configuration, mirroring AutoEncoderTrain / AutoEncoderBench --

BATCH_SIZE = 256
LATENT_DIM = 24
LEARNING_RATE = 3e-4
E_HIDDEN1 = 512
E_HIDDEN2 = 256
D_HIDDEN1 = 256
D_HIDDEN2 = 512
INPUT_SIZE = 28 * 28

BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


# -- Parameters, one node per case class in Autoencoder.Params --


class AffineParams(NamedTuple):
    weight: jax.Array
    bias: jax.Array


class EncoderParams(NamedTuple):
    layer1: AffineParams
    layer2: AffineParams
    latent_layer: AffineParams


class DecoderParams(NamedTuple):
    layer1: AffineParams
    layer2: AffineParams
    output_layer: AffineParams


class Params(NamedTuple):
    encoder: EncoderParams
    decoder: DecoderParams


class AdamState(NamedTuple):
    momentums: Params
    velocities: Params
    beta1t: jax.Array
    beta2t: jax.Array


class TrainState(NamedTuple):
    params: Params
    optimizer_state: AdamState
    last_cost: jax.Array


# -- Model: `y = xW + b`, exactly what deepwit's AffineLayer contracts to --


def affine(p: AffineParams, x: jax.Array) -> jax.Array:
    return x.dot(p.weight) + p.bias


def encode(p: EncoderParams, x: jax.Array) -> jax.Array:
    h1 = jax.nn.relu(affine(p.layer1, x))
    h2 = jax.nn.relu(affine(p.layer2, h1))
    return affine(p.latent_layer, h2)


def decode(p: DecoderParams, z: jax.Array) -> jax.Array:
    h1 = jax.nn.relu(affine(p.layer1, z))
    h2 = jax.nn.relu(affine(p.layer2, h1))
    return affine(p.output_layer, h2)


def logits(params: Params, x: jax.Array) -> jax.Array:
    return decode(params.decoder, encode(params.encoder, x))


def bce_from_logits(target: jax.Array, logit: jax.Array) -> jax.Array:
    """`max(z, 0) - z*y + log(1 + exp(-|z|))`, as deepwit's BinaryCrossEntropy writes it."""
    return jnp.maximum(logit, 0.0) - logit * target + jnp.log(1.0 + jnp.exp(-jnp.abs(logit)))


def cost_fn_for(samples: jax.Array) -> Callable[[Params], jax.Array]:
    def cost(params: Params) -> jax.Array:
        def per_sample(sample: jax.Array) -> jax.Array:
            original = sample.reshape(-1)
            return jax.vmap(bce_from_logits)(original, logits(params, original)).sum()

        return jax.vmap(per_sample)(samples).mean()

    return cost


# -- Adam, transcribed from dimwit.optimizer.Adam --


def adam_init(params: Params) -> AdamState:
    # Two distinct zero trees, as dimwit's `def zeros` gives: sharing one tree between
    # momentums and velocities would hand the donating JIT step the same buffer twice.
    def zeros() -> Params:
        return jax.tree.map(jnp.zeros_like, params)

    return AdamState(zeros(), zeros(), jnp.float32(1.0), jnp.float32(1.0))


def adam_update(grads: Params, params: Params, state: AdamState) -> tuple[Params, AdamState]:
    m = jax.tree.map(lambda m_prev, g: BETA1 * m_prev + (1.0 - BETA1) * g, state.momentums, grads)
    v = jax.tree.map(lambda v_prev, g: BETA2 * v_prev + (1.0 - BETA2) * g**2, state.velocities, grads)
    beta1t = state.beta1t * BETA1
    beta2t = state.beta2t * BETA2
    m_hat = jax.tree.map(lambda x: x / (1.0 - beta1t), m)
    v_hat = jax.tree.map(lambda x: x / (1.0 - beta2t), v)
    new_params = jax.tree.map(
        lambda p, mh, vh: p - (LEARNING_RATE * mh) / (jnp.sqrt(vh) + EPSILON), params, m_hat, v_hat
    )
    return new_params, AdamState(m, v, beta1t, beta2t)


def gradient_step(batch: jax.Array, state: TrainState) -> TrainState:
    cost, grads = jax.value_and_grad(cost_fn_for(batch))(state.params)
    new_params, new_optimizer_state = adam_update(grads, state.params, state.optimizer_state)
    return TrainState(new_params, new_optimizer_state, cost)


jit_gradient_step = jax.jit(gradient_step, donate_argnums=(1,))


# -- Data, read the same way MNISTLoader reads it --


def load_train_images(data_dir: Path) -> jax.Array:
    with open(data_dir / "train-images-idx3-ubyte", "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic: {magic}"
        assert (rows, cols) == (28, 28), f"Invalid image dimensions: {rows} x {cols}"
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
    return jnp.asarray(pixels, dtype=jnp.float32) / 255.0


def init_params(key: jax.Array) -> Params:
    """Glorot uniform with zero bias, which is what AffineLayer.Params.init does.

    Only used when no DeepWit-exported parameters are supplied; the benchmark loads
    those by preference so that both programs start from the same point.
    """

    def layer(k: jax.Array, fan_in: int, fan_out: int) -> AffineParams:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return AffineParams(
            weight=jax.random.uniform(k, (fan_in, fan_out), minval=-limit, maxval=limit),
            bias=jnp.zeros((fan_out,), dtype=jnp.float32),
        )

    keys = jax.random.split(key, 6)
    return Params(
        encoder=EncoderParams(
            layer(keys[0], INPUT_SIZE, E_HIDDEN1),
            layer(keys[1], E_HIDDEN1, E_HIDDEN2),
            layer(keys[2], E_HIDDEN2, LATENT_DIM),
        ),
        decoder=DecoderParams(
            layer(keys[3], LATENT_DIM, D_HIDDEN1),
            layer(keys[4], D_HIDDEN1, D_HIDDEN2),
            layer(keys[5], D_HIDDEN2, INPUT_SIZE),
        ),
    )


def load_exported_params(path: Path) -> Params:
    """Read the parameters DeepWit's TensorTreeIO pickled: nested tuples of numpy arrays."""
    with open(path, "rb") as f:
        (encoder, decoder) = pickle.load(f)

    def group(layers) -> list[AffineParams]:
        return [AffineParams(jnp.asarray(w), jnp.asarray(b)) for (w, b) in layers]

    return Params(encoder=EncoderParams(*group(encoder)), decoder=DecoderParams(*group(decoder)))


# -- Measurement, the same protocol as AutoEncoderBench.measure --


def measure(
    label: str,
    step: Callable[[jax.Array, TrainState], TrainState],
    batch: jax.Array,
    initial: Callable[[], TrainState],
    warmup: int,
    reps: int,
    steps_per_rep: int,
) -> dict:
    # A thunk, not a state: the donating JIT step deletes the buffers it is handed,
    # so every measurement has to be given a state of its own.
    state = jax.block_until_ready(initial())

    first_call_start = time.perf_counter()
    state = jax.block_until_ready(step(batch, state))
    first_call_ms = (time.perf_counter() - first_call_start) * 1e3

    for _ in range(warmup):
        state = step(batch, state)
    state = jax.block_until_ready(state)

    step_ms_per_rep = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(steps_per_rep):
            state = step(batch, state)
        state = jax.block_until_ready(state)
        step_ms_per_rep.append((time.perf_counter() - start) * 1e3 / steps_per_rep)

    assert not np.isnan(state.last_cost), f"{label} diverged to NaN"

    mean = statistics.fmean(step_ms_per_rep)
    return {
        "label": label,
        "first_call_ms": first_call_ms,
        "steps_per_rep": steps_per_rep,
        "reps": reps,
        "mean_step_ms": mean,
        "std_step_ms": statistics.stdev(step_ms_per_rep) if reps > 1 else 0.0,
        "median_step_ms": statistics.median(step_ms_per_rep),
        "min_step_ms": min(step_ms_per_rep),
        "max_step_ms": max(step_ms_per_rep),
        "steps_per_second": 1000.0 / mean,
        "step_ms_per_rep": step_ms_per_rep,
    }


def loss_trace(batch: jax.Array, initial: TrainState, steps: int) -> list[float]:
    state = initial
    trace = []
    for _ in range(steps):
        state = jit_gradient_step(batch, state)
        trace.append(float(state.last_cost))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="JIT steps per timed repetition")
    parser.add_argument("--eager-steps", type=int, default=20, help="eager steps per timed repetition")
    parser.add_argument("--reps", type=int, default=10, help="timed repetitions")
    parser.add_argument("--warmup", type=int, default=20, help="discarded steps before timing")
    parser.add_argument("--trace", type=int, default=20, help="steps of loss trace to record")
    parser.add_argument("--mode", choices=("both", "jit", "eager"), default="both")
    parser.add_argument("--out", type=Path, default=EXAMPLES_ROOT / "out" / "bench")
    parser.add_argument("--data", type=Path, default=EXAMPLES_ROOT / "data")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = jax.devices()[0]
    print(f"Device: {device}")
    print(f"JAX:    {jax.__version__}")

    images = load_train_images(args.data)
    batch = jax.block_until_ready(images[:BATCH_SIZE])

    exported = args.out / "init_params.pkl"
    if exported.exists():
        params = load_exported_params(exported)
        print(f"Parameters: loaded from {exported}")
    else:
        params = init_params(jax.random.key(42))
        print("Parameters: freshly initialized (no DeepWit export found)")

    def initial() -> TrainState:
        # A fresh copy each time: the donating JIT step deletes the buffers it is handed.
        fresh = jax.tree.map(lambda x: x.copy(), params)
        return TrainState(fresh, adam_init(fresh), jnp.float32(-1.0))

    batch_checksum = float(batch.sum())
    print(f"Batch checksum (sum of pixels): {batch_checksum:.6f}")

    # Measured before the loss trace, so that the first call of each configuration is a
    # genuinely cold compile rather than one served from a cache the trace has already filled.
    measurements = []
    if args.mode in ("both", "jit"):
        measurements.append(
            measure("jax-jit", jit_gradient_step, batch, initial, args.warmup, args.reps, args.steps)
        )
    if args.mode in ("both", "eager"):
        measurements.append(
            measure("jax-eager", gradient_step, batch, initial, args.warmup, args.reps, args.eager_steps)
        )

    print()
    header = f"{'configuration':<28} {'mean ms':>9}   {'std':<8} {'median':>9} {'min':>9} {'max':>9} {'steps/s':>9} {'1st call ms':>10}"
    print(header)
    for m in measurements:
        print(
            f"{m['label']:<28} {m['mean_step_ms']:9.3f} ± {m['std_step_ms']:<8.3f} "
            f"{m['median_step_ms']:9.3f} {m['min_step_ms']:9.3f} {m['max_step_ms']:9.3f} "
            f"{m['steps_per_second']:9.1f} {m['first_call_ms']:10.1f}"
        )

    trace = loss_trace(batch, jax.block_until_ready(initial()), args.trace)
    with open(args.out / "python_loss_trace.csv", "w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(trace, start=1):
            f.write(f"{i},{loss}\n")
    print(f"Loss trace: {trace[0]} .. {trace[-1]}")

    report = {
        "implementation": "jax-python",
        "device": str(device),
        "jax_version": jax.__version__,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": args.warmup,
        "batch_checksum": batch_checksum,
        "measurements": measurements,
    }
    with open(args.out / "python_bench.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out / 'python_bench.json'}")


if __name__ == "__main__":
    main()
