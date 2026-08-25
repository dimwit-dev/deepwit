"""The MNIST CNN benchmark in plain JAX, swept over batch size.

The counterpart to `MNistCNNBench.scala`. An independent check on the autoencoder result with
a different architecture, a different optimizer (plain gradient descent) and a much smaller
parameter tree.

Equivalence is structural: the model is written per-sample under `jax.vmap`, and each
convolution adds and removes a dummy batch axis, because that is exactly what dimwit's
`conv2d` does inside `zipvmap`. The measurement protocol is imported from `autoencoder_bench`,
so it is literally the same code as the other benchmarks.

Run, from the repository root:
    uv run --no-sync python examples/src/main/python/mnist_cnn_bench.py --batches 128,512,2048,8192
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from autoencoder_bench import EXAMPLES_ROOT, measure

LEARNING_RATE = 0.01
NUM_HIDDEN1 = 16
NUM_HIDDEN2 = 32
KERNEL = 3
NUM_CLASSES = 10


class ConvParams(NamedTuple):
    kernel: jax.Array
    bias: jax.Array


class AffineParams(NamedTuple):
    weight: jax.Array
    bias: jax.Array


class Params(NamedTuple):
    conv1: ConvParams
    conv2: ConvParams
    output: AffineParams


class TrainState(NamedTuple):
    params: Params
    optimizer_state: None
    last_cost: jax.Array


class Batch(NamedTuple):
    images: jax.Array
    labels: jax.Array


def conv(p: ConvParams, x: jax.Array) -> jax.Array:
    """`x.conv2d(kernel, stride = 2, padding = SAME) +! bias`, per sample.

    dimwit adds a dummy batch axis and squeezes it again, since it convolves one sample at a
    time under vmap; mirrored here so the traced graph has the same shape.
    """
    out = jax.lax.conv_general_dilated(
        x[None],
        p.kernel,
        window_strides=(2, 2),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return jnp.squeeze(out, axis=0) + p.bias


def logits(params: Params, image: jax.Array) -> jax.Array:
    x = image[..., None]  # appendAxis(Channel)
    hidden = jax.nn.relu(conv(params.conv1, x))
    pixel_embeddings = jax.nn.relu(conv(params.conv2, hidden))
    image_embedding = pixel_embeddings.reshape(-1)  # flatten / ravel, C order
    return image_embedding.dot(params.output.weight) + params.output.bias


def logsumexp(x: jax.Array) -> jax.Array:
    """As deepwit's CategoricalCrossEntropy writes it: max + log(sum(exp(x - max)))."""
    m = x.max()
    return m + jnp.log(jnp.exp(x - m).sum())


def cross_entropy_from_logits(label: jax.Array, lg: jax.Array) -> jax.Array:
    return logsumexp(lg) - lg[label]


def cost_fn_for(images: jax.Array, labels: jax.Array) -> Callable[[Params], jax.Array]:
    def cost(params: Params) -> jax.Array:
        def per_sample(image: jax.Array, label: jax.Array) -> jax.Array:
            return cross_entropy_from_logits(label, logits(params, image))

        return jax.vmap(per_sample)(images, labels).mean()

    return cost


def gradient_step(batch: Batch, state: TrainState) -> TrainState:
    cost, grads = jax.value_and_grad(cost_fn_for(batch.images, batch.labels))(state.params)
    new_params = jax.tree.map(lambda p, g: p - LEARNING_RATE * g, state.params, grads)
    return TrainState(new_params, None, cost)


jit_gradient_step = jax.jit(gradient_step, donate_argnums=(1,))


# -- Data, read the same way MNISTLoader reads it --


def load_train(data_dir: Path) -> tuple[jax.Array, jax.Array]:
    with open(data_dir / "train-images-idx3-ubyte", "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic: {magic}"
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
    with open(data_dir / "train-labels-idx1-ubyte", "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic for labels: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return jnp.asarray(pixels, dtype=jnp.float32) / 255.0, jnp.asarray(labels, dtype=jnp.int32)


def load_exported_params(path: Path) -> Params:
    with open(path, "rb") as f:
        (conv1, conv2, output) = pickle.load(f)
    return Params(
        conv1=ConvParams(jnp.asarray(conv1[0]), jnp.asarray(conv1[1])),
        conv2=ConvParams(jnp.asarray(conv2[0]), jnp.asarray(conv2[1])),
        output=AffineParams(jnp.asarray(output[0]), jnp.asarray(output[1])),
    )


def init_params(key: jax.Array) -> Params:
    """Xavier uniform over a flat (fanIn, fanOut) matrix, then reshaped to the kernel shape.

    This is what deepwit's `xavierUniformKernel` does: fanIn is kh*kw*inC and fanOut is just
    the output channel count, not kh*kw*outC. Only reached without a DeepWit export.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    def uniform(k, fan_in: int, fan_out: int, shape) -> jax.Array:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(k, shape, minval=-limit, maxval=limit)

    embedding = 7 * 7 * NUM_HIDDEN2
    return Params(
        conv1=ConvParams(
            uniform(k1, KERNEL * KERNEL * 1, NUM_HIDDEN1, (KERNEL, KERNEL, 1, NUM_HIDDEN1)),
            jnp.zeros((NUM_HIDDEN1,), dtype=jnp.float32),
        ),
        conv2=ConvParams(
            uniform(k2, KERNEL * KERNEL * NUM_HIDDEN1, NUM_HIDDEN2, (KERNEL, KERNEL, NUM_HIDDEN1, NUM_HIDDEN2)),
            jnp.zeros((NUM_HIDDEN2,), dtype=jnp.float32),
        ),
        output=AffineParams(
            uniform(k3, embedding, NUM_CLASSES, (embedding, NUM_CLASSES)),
            jnp.zeros((NUM_CLASSES,), dtype=jnp.float32),
        ),
    )


def loss_trace(batch: Batch, initial: TrainState, steps: int) -> list[float]:
    state = initial
    trace = []
    for _ in range(steps):
        state = jit_gradient_step(batch, state)
        trace.append(float(state.last_cost))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=str, default="128,512,2048,8192")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eager-steps", type=int, default=5)
    parser.add_argument("--eager-reps", type=int, default=8)
    parser.add_argument("--trace", type=int, default=20)
    parser.add_argument("--mode", choices=("both", "jit", "eager"), default="both")
    parser.add_argument("--out", type=Path, default=EXAMPLES_ROOT / "out" / "bench" / "cnn")
    parser.add_argument("--data", type=Path, default=EXAMPLES_ROOT / "data")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(b) for b in args.batches.split(",")]

    device = jax.devices()[0]
    print(f"Device: {device}")
    print(f"JAX:    {jax.__version__}")

    images, labels = load_train(args.data)

    exported = args.out / "cnn_init_params.pkl"
    if exported.exists():
        params = load_exported_params(exported)
        print(f"Parameters: loaded from {exported.name}")
    else:
        params = init_params(jax.random.key(42))
        print("Parameters: freshly initialized (no DeepWit export found)")

    leaves = len(jax.tree.leaves(params)) + 1  # + last_cost; the SGD state holds nothing
    print(f"Train state: {leaves} tensors")

    def initial() -> TrainState:
        fresh = jax.tree.map(lambda x: x.copy(), params)
        return TrainState(fresh, None, jnp.float32(-1.0))

    configs = []
    for batch_size in batch_sizes:
        batch = jax.block_until_ready(Batch(images[:batch_size], labels[:batch_size]))
        print(f"\n== batch {batch_size} ==")

        measurements = []
        if args.mode in ("both", "jit"):
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

    trace_batch = jax.block_until_ready(Batch(images[:128], labels[:128]))
    batch_checksum = float(trace_batch.images.sum())
    trace = loss_trace(trace_batch, jax.block_until_ready(initial()), args.trace)
    with open(args.out / "cnn_python_loss_trace.csv", "w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(trace, start=1):
            f.write(f"{i},{loss}\n")
    print(f"\nBatch checksum (sum of pixels, batch 128): {batch_checksum:.6f}")
    print(f"Loss trace: {trace[0]} .. {trace[-1]}")

    report = {
        "implementation": "jax-python",
        "model": "mnist-cnn",
        "device": str(device),
        "jax_version": jax.__version__,
        "warmup_steps": args.warmup,
        "leaves": leaves,
        "batch_checksum": batch_checksum,
        "configs": configs,
    }
    with open(args.out / "python_batch_bench.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out / 'python_batch_bench.json'}")


if __name__ == "__main__":
    main()
