"""Isolates where a DeepWit step's host time goes, by attributing it rather than assuming it.

One compiled JAX function is called three ways with the *same* arguments:

  1. from Python, with Python-resident arguments  — the JAX dispatch baseline;
  2. from Scala, with the same Python-resident arguments — adds only ScalaPy's call
     mechanism, and no tree work whatsoever;
  3. from Scala, marshalling through `TensorTree.toPyTree` / `fromPyTree` each call —
     adds dimwit's per-leaf tree traversal, which is what `Jit.toPyJit` does.

(2) - (1) is the cost of crossing the JVM/CPython boundary once per call.
(3) - (2) is the cost of rebuilding the tree. Whichever dominates is where the overhead is.

The function is a near-identity `tree_map`, so device work is negligible and what is left is
dispatch — the quantity actually in question. It is also structure-agnostic, so it accepts
dimwit's plain nested tuples and needs no Python-side model definition to stay in sync.
"""

from __future__ import annotations

import time

import jax


@jax.jit
def _step(state):
    return jax.tree.map(lambda x: x + 0.0, state)


def step(state):
    return _step(state)


def block(x) -> None:
    jax.block_until_ready(x)


def leaf_count(tree) -> int:
    return len(jax.tree.leaves(tree))


def time_python(state, steps: int) -> float:
    """Baseline: the identical loop, entirely inside Python."""
    s = state
    for _ in range(20):
        s = _step(s)
    jax.block_until_ready(s)

    start = time.perf_counter()
    for _ in range(steps):
        s = _step(s)
    jax.block_until_ready(s)
    return (time.perf_counter() - start) * 1e3 / steps
