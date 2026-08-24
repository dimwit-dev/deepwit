```scala mdoc:invisible
import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.{Adam, AdamState}

import deepwit.activation.gelu
import deepwit.base.{AffineFormLayer, AffineLayer}
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.loss.SquaredError

dimwit.initialize()

trait Batch derives Label
trait Feature derives Label
trait Embedding derives Label
```
# DeepWit

A deep learning library for Scala 3, built on [DimWit](https://github.com/dimwit-dev/dimwit),
a named tensor library.
DeepWit's philosophy is the code should read like the theory, which we illustrate here on a regression example:

### 1. The module: $f(x; \theta)$

In DeepWit, a model is a parameterized function `y = f(x; θ)` that requires an explicit parameter argument.
The model's parameters $\theta$ are bundled within an explicit `Params` type.

```scala mdoc:silent
// Explicit parameter type
case class Params(
    layer1: AffineLayer.Params[Feature, Embedding, Float32],
    layer2: AffineLayer.Params[Embedding, Embedding, Float32],
    output: AffineFormLayer.Params[Embedding, Float32]
)

// A model is a parameterized function, given the parameters it is function mapping a feature vector to a scalar.
class MLP(params: Params) extends (Tensor1[Feature, Float32] => Tensor0[Float32]):

  private val layer1 = AffineLayer(params.layer1)
  private val layer2 = AffineLayer(params.layer2)
  private val output = AffineFormLayer(params.output)

  def apply(x: Tensor1[Feature, Float32]): Tensor0[Float32] =
    output(gelu(layer2(gelu(layer1(x)))))
```

### 2. The cost function: $\mathcal{L}(\theta; \vec{x}, \vec{y}) \to \mathbb{R}$

In DeepWit, we define an explicit cost function that first takes data like the current batch `xs, ys` and then is a function that maps parameters `Params` to a scalar cost `Tensor0[Float32]`; here the mean square error between predictions and ground truths:

```scala mdoc:silent
def costFnFor(
  xs: Tensor2[Batch, Feature, Float32],
  ys: Tensor1[Batch, Float32]
)(params: Params): Tensor0[Float32] =
  val model = MLP(params)
  zipvmap(Axis[Batch])(xs, ys): (x, y) =>
    SquaredError(y, model(x))
  .mean
```

### 3. The train trajectory: $\theta_{t+1} = \theta_t - \eta \nabla\mathcal{L}(\theta_t)$

In DeepWit, we have (1) an explicit training state, (2) an explicit parameter initialization, (3) an explicit differentiation of the cost function, and (4) explicit gradients; reflecting theory in code. Lastly, (5) the train trajectory is a train state iterator from a start state.

```scala mdoc:invisible
val numIterations = 3_000
val hiddenSize = 32
val learningRate = 3e-3f
val initKey = Key(42)
val checkpointRoot = "out/Regression"

// Stands in for the example's batch stream over a noisy curve. `continually` takes its body by
// name and the trajectory is never consumed here, so the `???` is never forced.
val trainBatchStream: Iterator[(Tensor2[Batch, Feature, Float32], Tensor1[Batch, Float32])] =
  Iterator.continually(???)
```
```scala mdoc:silent
// (1) Train state of parameters and optimizer state.
case class TrainState(params: Params, optimizerState: AdamState[Params])

val optimizer = Adam(learningRate)

// (2) Inital parameters and initial optimizer state form the initial training state.
val initState =
  val initialParams =
    val (layer1Key, layer2Key, outputKey) = initKey.splitToTuple(3)
    val featureExtent = Axis[Feature] -> 1
    val hiddenExtent = Axis[Embedding] -> hiddenSize
    Params(
      layer1 = AffineLayer.Params.init(featureExtent, hiddenExtent, layer1Key),
      layer2 = AffineLayer.Params.init(hiddenExtent, hiddenExtent, layer2Key),
      output = AffineFormLayer.Params.init(hiddenExtent, outputKey)
    )
  TrainState(initialParams, optimizer.init(initialParams))

def gradientStep(xs: Tensor2[Batch, Feature, Float32], ys: Tensor1[Batch, Float32], state: TrainState): TrainState =
  // (3) Differentiate the cost function of the current batch.
  val dCost = Autodiff.grad(costFnFor(xs, ys))
  // (4) Calculate and apply the gradients at the current params to form the next state.
  val grads = dCost(state.params)
  val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
  TrainState(newParams, newOptimizerState)
val jitGradientStep = jitDonatingUnsafe(gradientStep)

// (5) A lazy iterator represents the entire trajectory over training states.
val trainTrajectory: Iterator[TrainState] = trainBatchStream.scanLeft(initState):
  case (state, (xs, ys)) =>
    jitGradientStep(xs, ys, state)
```

Training the model reduces to a termination condition on this iterator; here after `numIterations` updates. 
A model checkpointer serializes the final train state object.

```scala mdoc:compile-only
val finalState = trainTrajectory.drop(numIterations).next()

TensorTreeCheckpointer.newIn(checkpointRoot).save(finalState, numIterations)
```

## What the explicitness buys

TODO

## What's in `core`

DeepWit provides implementations for core deep learning modules with clear, strongly-typed boundaries. 
The user code composes these core modules into custom architectures given the use case.

| Package | Contents |
| --- | --- |
| `deepwit.base` | `LinearLayer`, `AffineLayer`, and their scalar-valued `Form` variants |
| `deepwit.cnn` | `LinearConv2DLayer`, `AffineConv2DLayer`, both transpose variants, `MaxPool2DLayer` |
| `deepwit.attention` | scaled-dot-product scores, full / causal / custom masking, multi-head (fused and unfused), self-attention, a readable reference implementation |
| `deepwit.transformer` | `TransformerBlock`, `CrossTransformerBlock` — the residual skeleton, with the mixers left open |
| `deepwit.embedder` | `VocabularyEmbedder` (with tied unembedding), `LearnedAbsolutePositionalInjector`, `ImageToPatchEmbedder`, `PositionalEncoding.sinusoidal2D` |
| `deepwit.normalization` | `LayerNorm`, `RMSNorm` |
| `deepwit.activation` | `sigmoid`, `relu`, `gelu`, `softmax` |
| `deepwit.loss` | `CategoricalCrossEntropy`, `BernoulliCrossEntropy`, `BinaryCrossEntropy`, `SquaredError`, `AbsoluteError`, `Huber` |
| `deepwit.init` | Xavier/Glorot normal and uniform, for matrices and vectors |
| `deepwit.regularization` | `Perturbation` — thinning (dropout) as a mutation of the weights that *read* a feature |
| `deepwit.optimizer` | `LearningRateSchedule` (constant, linear warmup, cosine decay), `LearningRateScheduler`, `clipGlobalNorm` |
| `deepwit.training` | `Monitor` (step, loss, throughput, learning rate), `tapEvery` |
| `deepwit.checkpointing` | `TensorTreeCheckpointer` — save and load any `TensorTree` by iteration |

## Examples

Each has a `train` main writing checkpoints, and an `eval`
main that reads the newest run back.

| Example | What it shows |
| --- | --- |
| [`regression`](examples/src/main/scala/deepwit/examples/regression/Regression.scala) | The tour above: an MLP on a noisy curve, train and eval in a single file |
| [`mnistClassification`](examples/src/main/scala/deepwit/examples/mnistClassification/) | Convolutional classifier on MNIST |
| [`autoencoder`](examples/src/main/scala/deepwit/examples/autoencoder/) | Encoder/decoder with transpose convolutions |
| [`neuralImage`](examples/src/main/scala/deepwit/examples/neuralImage/) | MLP that stores an image by mapping coordinates to pixels |
| [`gpt`](examples/src/main/scala/deepwit/examples/gpt/) | GPT-2 decoder trained on FineWeb |
| [`thinning`](examples/src/main/scala/deepwit/examples/thinning/) | Two-moons classifier; showcasing network thinning (functional replacement for dropout) |
