from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken


SEED = 1337

DATA_DIR = Path("/home/mebr/Documents/Scala/modded-nanogpt/data/fineweb10B")
TRAIN_PREFIX = "fineweb_train_"
VAL_PREFIX = "fineweb_val_"
SHARD_HEADER_BYTES = 1024

RUNNING_BATCH_SIZE = 64
EFFECTIVE_BATCH_SIZE = 512
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // RUNNING_BATCH_SIZE

LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = LEARNING_RATE / 10.0
WARMUP_STEPS = 1_000
LR_DECAY_STEPS = 20_000
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.1
GRADIENT_CLIP_NORM = 1.0

NUM_LAYERS = 12
VOCAB_SIZE = 50_304
BLOCK_SIZE = 1_024
EMBEDDING_SIZE = 768
NUM_HEADS = 12
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
MLP_HIDDEN_SIZE = 3_072
LAYER_NORM_EPS = 1e-12

PARAM_DTYPE = jnp.float32
COMPUTE_DTYPE = jnp.bfloat16

AttentionImplementation = Literal["xla", "cudnn"]

ATTENTION_IMPLEMENTATION: AttentionImplementation = cast(AttentionImplementation, os.environ.get(
	"GPT_ATTENTION_IMPLEMENTATION",
	"cudnn" if jax.default_backend() == "gpu" else "xla",
))

VALIDATION_TOKENS = 10_485_760
VALIDATION_INTERVAL = 1_000
NUM_BATCHES_PER_VALIDATION = max(1, (VALIDATION_TOKENS // BLOCK_SIZE) // RUNNING_BATCH_SIZE)

LOG_EVERY = 1
MAX_STEPS = 20_000

PROMPT_TEXT = "Here is my grandmother's secret recipe for the best chocolate chip cookies. Ingredients:"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
EOS_TOKEN_ID = 50_256


ArrayTree = dict[str, Any]

TOKENIZER = tiktoken.get_encoding("gpt2")


@dataclass
class Timer:
	decay: float = 0.01
	last_time: float = field(default_factory=time.perf_counter)
	running_average: float | None = None

	def tick(self) -> None:
		now = time.perf_counter()
		elapsed = now - self.last_time
		if self.running_average is None:
			self.running_average = elapsed
		else:
			self.running_average = self.running_average * self.decay + elapsed * (1.0 - self.decay)
		self.last_time = now

	@property
	def running_avg_seconds(self) -> float:
		return 0.0 if self.running_average is None else self.running_average


class FineWebBatchStream:
	def __init__(
		self,
		data_dir: Path,
		file_prefix: str,
		batch_size: int,
		context_length: int,
		seed: int,
	) -> None:
		self.files = sorted(path for path in data_dir.iterdir() if path.name.startswith(file_prefix))
		if not self.files:
			raise FileNotFoundError(f"No files found matching prefix '{file_prefix}' in {data_dir}")
		self.batch_size = batch_size
		self.context_length = context_length
		self.rng = np.random.default_rng(seed)
		self.shard_index = -1
		self.batches_remaining = 0
		self.current_data: np.memmap[Any, Any] | None = None
		self.sequence_offsets = np.arange(context_length + 1, dtype=np.int64)
		self._load_next_shard()

	def _load_next_shard(self) -> None:
		self.shard_index = (self.shard_index + 1) % len(self.files)
		shard_path = self.files[self.shard_index]
		self.current_data = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=SHARD_HEADER_BYTES)
		shard_tokens = int(self.current_data.shape[0])
		self.batches_remaining = max(1, shard_tokens // (self.batch_size * self.context_length))

	def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
		if self.batches_remaining <= 0:
			self._load_next_shard()
		assert self.current_data is not None
		max_start = int(self.current_data.shape[0]) - self.context_length - 1
		if max_start < 0:
			raise ValueError("Shard is smaller than the configured context length")
		start_indices = self.rng.integers(0, max_start + 1, size=self.batch_size, dtype=np.int64)
		token_indices = start_indices[:, None] + self.sequence_offsets[None, :]
		sequences = np.asarray(self.current_data[token_indices], dtype=np.int32)
		self.batches_remaining -= 1
		return sequences[:, :-1], sequences[:, 1:]

	def next_accumulation(self) -> tuple[np.ndarray, np.ndarray]:
		inputs = np.empty((ACCUMULATION_STEPS, self.batch_size, self.context_length), dtype=np.int32)
		targets = np.empty_like(inputs)
		for micro_step in range(ACCUMULATION_STEPS):
			micro_inputs, micro_targets = self.next_batch()
			inputs[micro_step] = micro_inputs
			targets[micro_step] = micro_targets
		return inputs, targets


def zeros_like_tree(tree: ArrayTree) -> ArrayTree:
	return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=PARAM_DTYPE), tree)


def cast_tree(tree: ArrayTree, dtype: jnp.dtype) -> ArrayTree:
	return jax.tree_util.tree_map(lambda x: x.astype(dtype), tree)


def tree_add(left: ArrayTree, right: ArrayTree) -> ArrayTree:
	return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def tree_scale(tree: ArrayTree, scale: float | jax.Array) -> ArrayTree:
	return jax.tree_util.tree_map(lambda x: x * scale, tree)


def clip_by_global_norm(tree: ArrayTree, max_norm: float) -> tuple[ArrayTree, jax.Array]:
	sq_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree))
	global_norm = jnp.sqrt(sq_norm)
	scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
	return tree_scale(tree, scale), global_norm


def split_key(key: jax.Array, count: int) -> tuple[jax.Array, list[jax.Array]]:
	keys = jax.random.split(key, count + 1)
	return keys[0], list(keys[1:])


def init_layer_norm() -> ArrayTree:
	return {
		"weight": jnp.ones((EMBEDDING_SIZE,), dtype=PARAM_DTYPE),
		"bias": jnp.zeros((EMBEDDING_SIZE,), dtype=PARAM_DTYPE),
	}


def init_linear(key: jax.Array, fan_in: int, fan_out: int, std: float) -> ArrayTree:
	return {
		"weight": jax.random.normal(key, (fan_in, fan_out), dtype=PARAM_DTYPE) * std,
		"bias": jnp.zeros((fan_out,), dtype=PARAM_DTYPE),
	}


def init_block(key: jax.Array) -> ArrayTree:
	residual_std = 0.02 / math.sqrt(2.0 * NUM_LAYERS)
	key, (attn_key, proj_key, fc_key, mlp_proj_key) = split_key(key, 4)
	del key
	return {
		"ln_1": init_layer_norm(),
		"attn": {
			"c_attn": init_linear(attn_key, EMBEDDING_SIZE, 3 * EMBEDDING_SIZE, 0.02),
			"c_proj": init_linear(proj_key, EMBEDDING_SIZE, EMBEDDING_SIZE, residual_std),
		},
		"ln_2": init_layer_norm(),
		"mlp": {
			"c_fc": init_linear(fc_key, EMBEDDING_SIZE, MLP_HIDDEN_SIZE, 0.02),
			"c_proj": init_linear(mlp_proj_key, MLP_HIDDEN_SIZE, EMBEDDING_SIZE, residual_std),
		},
	}


def init_params(key: jax.Array) -> ArrayTree:
	key, subkeys = split_key(key, NUM_LAYERS + 3)
	del key
	blocks = tuple(init_block(block_key) for block_key in subkeys[:NUM_LAYERS])
	return {
		"wte": jax.random.normal(subkeys[NUM_LAYERS], (VOCAB_SIZE, EMBEDDING_SIZE), dtype=PARAM_DTYPE) * 0.02,
		"wpe": jax.random.normal(subkeys[NUM_LAYERS + 1], (BLOCK_SIZE, EMBEDDING_SIZE), dtype=PARAM_DTYPE) * 0.02,
		"blocks": blocks,
		"ln_f": init_layer_norm(),
	}


def linear(x: jax.Array, params: ArrayTree) -> jax.Array:
	weight = params["weight"].astype(x.dtype)
	bias = params["bias"].astype(x.dtype)
	return jnp.einsum("...c,cd->...d", x, weight) + bias


def layer_norm(x: jax.Array, params: ArrayTree) -> jax.Array:
	x_f32 = x.astype(jnp.float32)
	mean = jnp.mean(x_f32, axis=-1, keepdims=True)
	variance = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
	normalized = (x_f32 - mean) * jax.lax.rsqrt(variance + LAYER_NORM_EPS)
	normalized = normalized.astype(x.dtype)
	return normalized * params["weight"].astype(x.dtype) + params["bias"].astype(x.dtype)


def causal_self_attention(x: jax.Array, params: ArrayTree) -> jax.Array:
	batch_size, sequence_length, _ = x.shape
	qkv = linear(x, params["c_attn"])
	q, k, v = jnp.split(qkv, 3, axis=-1)

	q = q.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_SIZE)
	k = k.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_SIZE)
	v = v.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_SIZE)

	attention_output = jax.nn.dot_product_attention(
		q,
		k,
		v,
		is_causal=True,
		implementation=ATTENTION_IMPLEMENTATION,
	)
	attention_output = attention_output.reshape(batch_size, sequence_length, EMBEDDING_SIZE)
	return linear(attention_output, params["c_proj"])


def mlp(x: jax.Array, params: ArrayTree) -> jax.Array:
	hidden = linear(x, params["c_fc"])
	hidden = jax.nn.gelu(hidden)
	return linear(hidden, params["c_proj"])


def gpt_forward(params: ArrayTree, token_context: jax.Array) -> jax.Array:
	_, sequence_length = token_context.shape
	token_embeddings = params["wte"][token_context]
	position_embeddings = params["wpe"][jnp.arange(sequence_length)]
	x = token_embeddings + position_embeddings[None, :, :].astype(token_embeddings.dtype)
	for block in params["blocks"]:
		x = x + causal_self_attention(layer_norm(x, block["ln_1"]), block["attn"])
		x = x + mlp(layer_norm(x, block["ln_2"]), block["mlp"])
	x = layer_norm(x, params["ln_f"])
	return jnp.einsum("btc,vc->btv", x, params["wte"].astype(x.dtype))


def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
	log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
	target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
	return -jnp.mean(target_log_probs)


def batch_loss(params: ArrayTree, inputs: jax.Array, targets: jax.Array) -> jax.Array:
	logits = gpt_forward(params, inputs)
	return cross_entropy_loss(logits, targets)


def learning_rate_for_step(step: int) -> float:
	if step < WARMUP_STEPS:
		return LEARNING_RATE * float(step + 1) / float(WARMUP_STEPS)
	if step >= WARMUP_STEPS + LR_DECAY_STEPS:
		return MIN_LEARNING_RATE
	decay_step = step - WARMUP_STEPS
	decay_ratio = decay_step / LR_DECAY_STEPS
	cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


def init_train_state(params: ArrayTree) -> dict[str, Any]:
	return {
		"params": params,
		"m": zeros_like_tree(params),
		"v": zeros_like_tree(params),
		"step": jnp.array(0, dtype=jnp.int32),
	}


def _microbatch_gradients(params: ArrayTree, inputs: jax.Array, targets: jax.Array) -> tuple[jax.Array, ArrayTree]:
	params_compute = cast_tree(params, COMPUTE_DTYPE)
	loss, grads = jax.value_and_grad(batch_loss)(params_compute, inputs, targets)
	return loss.astype(PARAM_DTYPE), cast_tree(grads, PARAM_DTYPE)


def _apply_gradients(
	state: dict[str, Any],
	grads: ArrayTree,
	learning_rate: jax.Array,
	loss: jax.Array,
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
	clipped_grads, grad_norm = clip_by_global_norm(grads, GRADIENT_CLIP_NORM)

	next_step = state["step"] + 1
	next_m = jax.tree_util.tree_map(
		lambda m, g: ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * g,
		state["m"],
		clipped_grads,
	)
	next_v = jax.tree_util.tree_map(
		lambda v, g: ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * jnp.square(g),
		state["v"],
		clipped_grads,
	)

	bias_correction1 = 1.0 - jnp.power(jnp.array(ADAM_BETA1, dtype=PARAM_DTYPE), next_step.astype(PARAM_DTYPE))
	bias_correction2 = 1.0 - jnp.power(jnp.array(ADAM_BETA2, dtype=PARAM_DTYPE), next_step.astype(PARAM_DTYPE))

	next_params = jax.tree_util.tree_map(
		lambda p, m, v: p - learning_rate * ((m / bias_correction1) / (jnp.sqrt(v / bias_correction2) + ADAM_EPS) + WEIGHT_DECAY * p),
		state["params"],
		next_m,
		next_v,
	)

	next_state = {
		"params": next_params,
		"m": next_m,
		"v": next_v,
		"step": next_step,
	}
	metrics = {
		"loss": loss,
		"grad_norm": grad_norm,
	}
	return next_state, metrics


def _eval_step(params: ArrayTree, inputs: jax.Array, targets: jax.Array) -> jax.Array:
	params_compute = cast_tree(params, COMPUTE_DTYPE)
	return batch_loss(params_compute, inputs, targets)


microbatch_gradients = jax.jit(_microbatch_gradients)
apply_gradients = _apply_gradients # # No JIT here for fairer comparision with DimWit (--> almost no difference)
eval_step = jax.jit(_eval_step)


def count_parameters(params: ArrayTree) -> int:
	return sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params))


def generate(
	params: ArrayTree,
	prompt: str,
	max_new_tokens: int = MAX_NEW_TOKENS,
	temperature: float = TEMPERATURE,
	seed: int = SEED,
) -> str:
	key = jax.random.PRNGKey(seed)
	token_ids = TOKENIZER.encode(prompt)
	params_compute = cast_tree(params, COMPUTE_DTYPE)
	for _ in range(max_new_tokens):
		context = token_ids[-BLOCK_SIZE:]
		context_array = jnp.asarray(context, dtype=jnp.int32)[None, :]
		logits = gpt_forward(params_compute, context_array)
		next_token_logits = logits[0, len(context) - 1].astype(jnp.float32) / temperature
		key, sample_key = jax.random.split(key)
		next_token = int(jax.random.categorical(sample_key, next_token_logits))
		token_ids.append(next_token)
		if next_token == EOS_TOKEN_ID:
			break
	return TOKENIZER.decode(token_ids)


def validate(params: ArrayTree, val_stream: FineWebBatchStream) -> float:
	losses = []
	for _ in range(NUM_BATCHES_PER_VALIDATION):
		inputs, targets = val_stream.next_batch()
		batch_loss_value = eval_step(
			params,
			jnp.asarray(inputs, dtype=jnp.int32),
			jnp.asarray(targets, dtype=jnp.int32),
		)
		losses.append(float(batch_loss_value))
	return float(sum(losses) / len(losses))


def train() -> dict[str, Any]:
	if EFFECTIVE_BATCH_SIZE % RUNNING_BATCH_SIZE != 0:
		raise ValueError("EFFECTIVE_BATCH_SIZE must be divisible by RUNNING_BATCH_SIZE")

	params = init_params(jax.random.PRNGKey(SEED))
	print(f"Using JAX devices: {jax.devices()}")
	print(f"Model parameters: {count_parameters(params):,}")
	print(f"Training data: {DATA_DIR}")
	print("Training...")

	train_stream = FineWebBatchStream(DATA_DIR, TRAIN_PREFIX, RUNNING_BATCH_SIZE, BLOCK_SIZE, SEED)
	val_stream = FineWebBatchStream(DATA_DIR, VAL_PREFIX, RUNNING_BATCH_SIZE, BLOCK_SIZE, SEED + 1)

	state = init_train_state(params)
	timer = Timer()

	for _ in range(MAX_STEPS):
		learning_rate = learning_rate_for_step(int(jax.device_get(state["step"])))
		loss_sum = jnp.array(0.0, dtype=PARAM_DTYPE)
		grad_sum = zeros_like_tree(state["params"])

		for _micro_step in range(ACCUMULATION_STEPS):
			host_inputs, host_targets = train_stream.next_batch()
			micro_loss, micro_grads = microbatch_gradients(
				state["params"],
				jnp.asarray(host_inputs, dtype=jnp.int32),
				jnp.asarray(host_targets, dtype=jnp.int32),
			)
			loss_sum = loss_sum + micro_loss
			grad_sum = tree_add(grad_sum, micro_grads)

		mean_loss = loss_sum / ACCUMULATION_STEPS
		mean_grads = tree_scale(grad_sum, 1.0 / ACCUMULATION_STEPS)
		state, metrics = apply_gradients(
			state,
			mean_grads,
			jnp.asarray(learning_rate, dtype=PARAM_DTYPE),
			mean_loss,
		)

		current_step = int(jax.device_get(state["step"]))
		if current_step % LOG_EVERY == 0:
			timer.tick()
			seconds_per_batch = max(timer.running_avg_seconds, 1e-9)
			timestamp = datetime.now(timezone.utc).isoformat()
			log_data = {
				"timestamp": timestamp,
				"iter": current_step,
				"tokens_per_s": f"{(EFFECTIVE_BATCH_SIZE * BLOCK_SIZE) / seconds_per_batch:.2f}",
				"samples_per_s": f"{EFFECTIVE_BATCH_SIZE / seconds_per_batch:.2f}",
				"s_per_batch": f"{seconds_per_batch:.2f}",
				"learning_rate": f"{learning_rate:.8f}",
				"step_cost": f"{float(metrics['loss']):.2f}",
			}
			print(" | ".join(f"{key}: {value}" for key, value in log_data.items()))

		if current_step % VALIDATION_INTERVAL == 0:
			print("-" * 30)
			print(f"Performing validation at iter {current_step}...")
			validation_cost = validate(state["params"], val_stream)
			print(f"Validation cost {current_step}: {validation_cost:.4f}")
			print("-" * 30)

	return state


def main() -> None:
	state = train()
	sample = generate(state["params"], PROMPT_TEXT)
	print("\nPrompt:", PROMPT_TEXT)
	print("Response:", sample)


if __name__ == "__main__":
	main()
