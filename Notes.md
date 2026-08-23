# DeepWit — Notes

Working notes on the state of the library: what `core` contains today, what is still missing
before it is a complete deep learning library, and which examples are still missing.

Last surveyed: 2026-08-20 (commit `26c00d9`).

## Scope: DeepWit vs. DimWit

Everything tensor-shaped lives in DimWit and is *not* a DeepWit gap:
tensors and labelled axes, `Autodiff.{grad, valueAndGrad, jacobian, hessian}`, `jit` /
`jitDonatingUnsafe`, `vmap` / `zipvmap`, random keys and distributions, `TensorTree` +
`TensorTreeIO`, and the gradient optimizers (`GradientDescent`, `Adam`, `AdamW`, `Lion`).
DeepWit is the layer above: parameterized modules, initialization, losses, regularization,
schedules, and the vocabulary a training loop is written in.

## Snapshot: what core has today

| Area | Present |
| --- | --- |
| `base` | `LinearLayer`, `AffineLayer`, `LinearFormLayer`, `AffineFormLayer` |
| `cnn` | `LinearConv2DLayer`, `AffineConv2DLayer`, both transpose variants, `MaxPool2DLayer` |
| `attention` | scaled-dot-product score, full / causal / custom attention, multi-head (fused + unfused), self-attention, reference implementation |
| `transformer` | `TransformerBlock`, `CrossTransformerBlock` |
| `embedder` | `VocabularyEmbedder` (with tied unembedding), `LearnedAbsolutePositionalInjector`, `ImageToPatchEmbedder`, `PositionalEncoding.sinusoidal2D` |
| `normalization` | `LayerNorm`, `RMSNorm` |
| `activation` | re-export of DimWit's `sigmoid`, `relu`, `gelu`, `softmax` |
| `loss` | `CategoricalCrossEntropy`, `BernoulliCrossEntropy`, `BinaryCrossEntropy` |
| `init` | Xavier/Glorot normal + uniform (matrix and vector) |
| `regularization` | `Perturbation` (thinning: dropout as a mutation of the weights that read a feature) |
| `optimizer` | `LearningRateSchedule` + `LearningRateScheduler`, constant/linear-warmup/cosine-decay, `clipGlobalNorm` |
| `training` | `Monitor` (step, loss, throughput, LR), `tapEvery` |
| `checkpointing` | `TensorTreeCheckpointer` (pickle save/load by iteration) |

---

# Part 1 — Missing in core

Ordered by how often the absence forces a user to hand-roll something in their training loop.

## P0 — Gaps hit by nearly every model

### Regression and distributional losses
`loss` only covers cross entropy. Missing: mean squared error, mean absolute error, Huber /
smooth-L1, Gaussian negative log-likelihood, KL divergence, and cosine embedding loss. Without
MSE, no regression model, autoencoder with continuous targets, or diffusion model can be written
without inlining the arithmetic. KL divergence additionally blocks VAEs and distillation.

### Metrics
There is no `metric` package at all. Accuracy is currently recomputed by hand in
[MNistCNNEval.scala:54](examples/src/main/scala/deepwit/examples/mnistClassification/MNistCNNEval.scala#L54).
Needed: accuracy, top-k accuracy, perplexity, precision/recall/F1, confusion matrix, and a
running-average aggregator so a metric can be folded over a validation stream rather than
materialized in one batch.

### Initialization schemes
Only Xavier/Glorot exists, which is the wrong default for the ReLU/GELU networks the library
already ships modules for. Missing: He/Kaiming (normal + uniform), LeCun, truncated normal
(what GPT-2-style embeddings want, σ = 0.02), orthogonal, constant/zeros/ones, and the
residual-scaled variant (σ / √(2 · numLayers)) that deep transformer stacks need for stable
early training. Also missing: fan-in/fan-out computation as a reusable concept — `cnn` currently
derives it privately in [package.scala](core/src/main/scala/deepwit/cnn/package.scala).

### Checkpoint restore and resume
`TensorTreeCheckpointer` can save and can load a specific iteration, but there is no `latest`,
no resume-the-training-state path, and `load` still wraps in `Some(...)` with a TODO because
`TensorTreeIO.load` cannot fail gracefully. The consequence is visible in
[GPTGen.scala:24](examples/src/main/scala/deepwit/examples/gpt/GPTGen.scala#L24), where loading a
trained model is literally `???`. Also missing: retention policy (keep last N / keep best),
saving metadata alongside the tree (step, config, loss), and a checkpoint format that survives
a change in the parameter tree's shape.

### Metric logging
There is no logging sink — `TenZarrLogger` existed once and was removed. `Monitor` only renders
a line to stdout, so nothing a run produces is queryable afterwards. Needed: an append-only
scalar log (Zarr or JSONL) with a plotting-friendly layout, and ideally a TensorBoard-compatible
writer.

## P1 — Gaps that block whole model families

### Recurrent modules
No RNN, LSTM, GRU, or scan-over-time primitive. Beyond the models themselves, the missing piece
is the general one: a `scan` abstraction for carrying state across a sequence axis, which is
also what an ODE solver or a diffusion sampler wants.

### Normalization beyond the per-sample kind
`LayerNorm` and `RMSNorm` both normalize within one sample, which is what transformers need.
Missing: `BatchNorm` (and with it, the general problem of a module that carries running
statistics — a genuine design question given the explicit-state philosophy: the running mean and
variance are state that is neither a parameter nor a hyperparameter), `GroupNorm`, and
`InstanceNorm`. Convnets and GANs are hard to reproduce without them.

### Positional encodings for sequences
`PositionalEncoding` only offers `sinusoidal2D`, for images. Missing: 1D sinusoidal encoding,
rotary position embeddings (RoPE), and ALiBi. RoPE in particular is table stakes for any modern
transformer, and the GPT example is stuck on learned absolute positions because of it.

### Inference-time attention
No KV cache. [GPT.generate](examples/src/main/scala/deepwit/examples/gpt/GPT.scala#L34) re-runs the
full forward pass over the whole context per token, which is O(n²) per sequence. A cache is
also the interesting design problem here: it is exactly the kind of hidden mutable state the
library refuses elsewhere, so it wants an explicit carried-state formulation.

### Sampling / decoding utilities
Temperature scaling and categorical sampling are inlined in the GPT example. Missing from core:
top-k, top-p (nucleus), min-p, repetition penalty, greedy/argmax decode, and beam search.

### Pooling and convolution coverage
Only `MaxPool2D`. Missing: average pooling, global/adaptive pooling, 1D convolution (audio and
text), dilated and grouped/depthwise convolution, and upsampling/interpolation layers (which
a U-Net needs alongside the transpose convolutions that already exist).

## P2 — Quality-of-life for the training loop

### Regularization
Only feature thinning, via `Perturbation.thin`. The parameter-perturbation formulation extends
naturally to DropConnect (`thinWeights`), spatial dropout (`thinChannels`), Gaussian noise and
additive weight noise, none written yet.
It does not extend to a feature consumed by anything non-linear, so attention dropout and dropout
inside a residual stream still need the activation-space form. Missing: attention dropout wired
through the attention modules,
stochastic depth / DropPath (needed for deep ViTs), label smoothing, and mixup/cutmix as data
augmentation. Weight decay exists, but only inside DimWit's `AdamW`.

### Gradient handling
`clipGlobalNorm` exists; `clipByValue` does not. Gradient accumulation is hand-rolled with a
`foldLeft` over a `List[BatchSample]` in
[GPTTrain.scala:126-133](examples/src/main/scala/deepwit/examples/gpt/GPTTrain.scala#L126-L133) —
that pattern is general enough to belong in `training`. Also missing: a parameter EMA (needed by
diffusion training and by most reported ImageNet numbers), and gradient/activation norm
diagnostics for debugging a diverging run.

### Optimizer coverage
DimWit supplies SGD, Adam, AdamW, and Lion. Missing across both projects: momentum/Nesterov SGD,
RMSProp, and — more importantly for DeepWit — per-parameter-group configuration, i.e. the
standard "no weight decay on biases and norm gains" rule, which today cannot be expressed at all.

### Learning rate schedules
Constant, linear warmup, and cosine decay, composable via `followBy` and `delay`. Missing: step
decay, exponential decay, inverse-sqrt (the original transformer schedule), one-cycle, and
plateau-triggered decay (which needs a schedule that reads training state, not just the step).

### Data pipeline
There is no `data` package. Every example hand-rolls its batching — `MNISTDataset.toBatchStream`
in the examples, `FineWebDataset.batchStream` in the GPT example. The reusable concepts are:
a dataset abstraction over a labelled sample axis, shuffling driven by an explicit random key,
train/validation splitting, prefetching, and epoch semantics (the current MNIST stream wraps
modulo the dataset size and has no notion of an epoch boundary).

### Mixed precision
The GPT example casts parameters to `BFloat16` by hand for the backward pass and back to
`Float32` for the update, and divides accumulated gradients manually. A precision policy —
compute dtype vs. parameter dtype vs. accumulation dtype — plus loss scaling for `Float16`
belongs in core.

### Module composition
Modules compose today with plain `andThen` on `Function1`, which is philosophically right, but
there is no vocabulary for the recurring shapes: a sequential stack whose parameters are a
`List` (`GPT.Params.init(numTransformerLayers = ...)` builds one ad hoc), residual wrapping, or
a per-layer parameter tree that initializes and maps uniformly.

### Weight tying and parameter sharing
`VocabularyEmbedder.unembed` ties the embedding to the output projection implicitly. There is no
general story for sharing one parameter tensor between two modules, and none for freezing a
subtree of parameters (transfer learning, fine-tuning, LoRA-style adapters).

## P3 — Ecosystem

- **Tokenization.** The GPT example shells out to Python's `tiktoken`. A native BPE encoder (or
  at least a documented, supported bridge) is the missing piece for any text example to stand alone.
- **Model zoo / weight import.** No way to load HuggingFace or PyTorch checkpoints, which makes
  every pretrained-model example impossible.
- **Multi-device.** DimWit has a `hardware.Device`, but DeepWit has no data-parallel or sharded
  training story.
- **Docs.** [docs/README.md](docs/README.md) ends in a literal `TODO`; the "Downstream Benefits"
  section has one of its planned highlights written. Missing: an API guide per package, and a
  from-scratch tutorial that goes further than the design philosophy.
- **Test coverage.** Every core module has a suite except `transformer` (both `TransformerBlock`
  and `CrossTransformerBlock` are untested since the `TransformerLayer` → `TransformerBlock`
  rename dropped the old suites), `checkpointing` beyond the round-trip, and `Monitor`'s
  throughput arithmetic.

---

# Part 2 — Missing examples

Present today: MNIST CNN classification (train + eval), MNIST autoencoder (train + eval),
GPT-2 on FineWeb (train + generate), and dropout-as-thinning on two moons (train + eval). All three share the MNIST/FineWeb loaders and the
`scanLeft`-over-a-batch-stream training loop idiom.

## Should exist first — they exercise the P0/P1 gaps

1. **Linear / logistic regression from scratch.** The smallest possible example, and currently
   absent: the entry point that shows `Autodiff.grad` + an optimizer + an explicit parameter
   object with nothing else in the way. Needs MSE.
2. **MLP on MNIST.** The obvious rung between regression and the CNN, and the natural place to
   show explicit dropout and a learning rate schedule on something that trains in seconds.
3. **Restore the diffusion example.** `MNISTDiffusion` existed and was dropped (its checkpoints
   are still in `examples/out/DiffusionMNIST/`). It is the best driver for MSE loss, parameter
   EMA, a noise schedule, and a sampling loop, and it is the one example whose absence is
   already visible in the repository.
4. **Vision Transformer on MNIST/CIFAR.** `ImageToPatchEmbedder` and `TransformerBlock` both
   exist and no example uses them together — the ViT path is currently untested end to end.
5. **Sequence model on a small corpus (char-level RNN/LSTM).** Blocked on the recurrent modules;
   worth listing because it is what would validate the `scan` design.

## Rounding out the model families

6. **Variational autoencoder.** The natural sequel to the existing autoencoder, and the example
   that motivates KL divergence and the reparameterization trick — the latter being a good
   showcase for explicit random keys.
7. **U-Net segmentation.** Uses transpose convolutions (implemented, unused by any example) and
   would force the missing upsampling/skip-connection story.
8. **Seq2seq translation with cross-attention.** `CrossTransformerBlock` exists and no example
   uses it; an encoder-decoder example is the only thing that would exercise it.
9. **GAN.** Two parameter trees, two optimizers, alternating updates — a good stress test of
   whether the explicit-parameters philosophy holds up under a non-trivial training loop.
10. **Fine-tuning a pretrained model.** Blocked on weight import and parameter freezing;
    the example that would prove the library is usable for real work rather than teaching.

## Examples about the library rather than a model

11. **Checkpoint + resume.** Kill a run, restart it, show the loss curve continue. Blocked on
    checkpoint restore; would also fix `GPTGen`'s `???`.
12. **Gradient accumulation and mixed precision, in isolation.** Both currently exist only
    tangled into `GPTTrain`, which is the largest and least approachable example.
13. **Custom module walkthrough.** Write a new layer — params object, initializer, `Function1`
    instance, test — as the documentation of the module contract described in
    [docs/README.md](docs/README.md).
14. **Notebook / interactive examples.** `.sc` scripts exist for the autoencoder and CNN eval,
    but they are undocumented and not part of any build target.
