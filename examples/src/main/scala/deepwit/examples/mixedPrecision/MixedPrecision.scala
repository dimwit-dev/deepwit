package deepwit.examples.mixedPrecision

import dimwit.*
import dimwit.optimizer.{Adam, AdamState}
import dimwit.stats.Uniform
import dimwit.TreeOf.ops.asFloats

import deepwit.training.after
import deepwit.activation.gelu
import deepwit.base.{AffineFormLayer, AffineLayer}
import deepwit.loss.SquaredError
import deepwit.optimizer.{LossScale, allFinite, select}

trait Sample derives Label
trait Feature derives Label
trait Embedding derives Label

class MLP[V: IsFloating](params: MLP.Params[V]) extends (Tensor1[Feature, V] => Tensor0[V]):

  private val layer1 = AffineLayer(params.layer1)
  private val layer2 = AffineLayer(params.layer2)
  private val output = AffineFormLayer(params.output)

  def apply(x: Tensor1[Feature, V]): Tensor0[V] = output(gelu(layer2(gelu(layer1(x)))))

object MLP:

  case class Params[V](
      layer1: AffineLayer.Params[Feature, Embedding, V],
      layer2: AffineLayer.Params[Embedding, Embedding, V],
      output: AffineFormLayer.Params[Embedding, V]
  )

  object Params:

    def init(hiddenSize: Int, key: Key): Params[Float32] =
      val (layer1Key, layer2Key, outputKey) = key.splitToTuple(3)
      val hiddenExtent = Axis[Embedding] -> hiddenSize
      Params(
        layer1 = AffineLayer.Params.init(Axis[Feature] -> 1, hiddenExtent, layer1Key),
        layer2 = AffineLayer.Params.init(hiddenExtent, hiddenExtent, layer2Key),
        output = AffineFormLayer.Params.init(hiddenExtent, outputKey)
      )

case class TrainState(
    params: MLP.Params[Float32],
    optimizerState: AdamState[MLP.Params[Float32]],
    lossScale: LossScale,
    skippedSteps: Tensor0[Int32]
)

/** Fits a curve of amplitude 1e-3 in Float32, in Float16, and in Float16 with a [[LossScale]].
  *
  * The small amplitude makes the gradients small. Float16 rounds them to zero and barely learns; with the
  * loss scale it matches Float32. Predicting zero everywhere would give an RMSE of about 0.71 amplitudes.
  */
@main
def mixedPrecision(): Unit =

  val numIterations = 3_000
  val numSamples = 256
  val hiddenSize = 32
  val learningRate = 3e-3f
  val amplitude = 1e-3f

  val (dataKey, initKey) = Key(42).split2()
  val shape = Shape(Axis[Sample] -> numSamples, Axis[Feature] -> 1)
  val xs = Uniform(Tensor(shape).fill(-1f), Tensor(shape).fill(1f)).sample(dataKey)
  val ys = (xs.slice(Axis[Feature].at(0)) *! (2f * math.Pi.toFloat)).sin *! amplitude

  /** Runs the model in `vtype` and the loss in Float32. The master weights stay Float32, and so do their gradients. */
  def costIn[V: IsFloating](vtype: VType[V])(params: MLP.Params[Float32]): Tensor0[Float32] =
    val model = MLP(params.asFloats(vtype))
    zipvmap(Axis[Sample])(xs.asFloat(vtype), ys): (x, y) =>
      val prediction = model(x) * amplitude
      SquaredError(y, prediction.asFloat(VType[Float32]))
    .mean

  val optimizer = Adam(learningRate)

  def plainStep[V: IsFloating](vtype: VType[V])(state: TrainState): TrainState =
    val grads = Autodiff.grad(costIn(vtype))(state.params)
    val (params, optimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    state.copy(params = params, optimizerState = optimizerState)

  def scaledStep(state: TrainState): TrainState =
    val lossScale = state.lossScale
    val grads = lossScale.unscaled(Autodiff.grad((params: MLP.Params[Float32]) => lossScale.scaled(costIn(VType[Float16])(params)))(state.params))
    val (params, optimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    val finite = allFinite(grads)
    TrainState(
      select(finite, params, state.params),
      select(finite, optimizerState, state.optimizerState),
      lossScale.next(finite, growthInterval = 200),
      where(finite, state.skippedSteps, state.skippedSteps + 1)
    )

  val initialState =
    val params = MLP.Params.init(hiddenSize, initKey)
    TrainState(params, optimizer.init(params), LossScale.initial(), Tensor0(0))

  def run(name: String, step: TrainState => TrainState, scaled: Boolean = false): Unit =
    val jitStep = jitDonatingUnsafe(step)
    val state = Iterator.iterate(initialState)(jitStep).after(numIterations)
    val rmse = math.sqrt(costIn(VType[Float32])(state.params).item) / amplitude
    val scaleInfo = if scaled then f"   final scale: 2^${math.log(state.lossScale.scale.item) / math.log(2)}%.0f   skipped steps: ${state.skippedSteps.item}" else ""
    println(f"$name%-20s RMSE / amplitude: $rmse%.4f$scaleInfo")

  run("Float32", plainStep(VType[Float32]))
  run("Float16", plainStep(VType[Float16]))
  run("Float16 + LossScale", scaledStep, scaled = true)
