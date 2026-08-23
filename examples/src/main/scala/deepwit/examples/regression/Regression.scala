package deepwit.examples.regression

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.{Adam, AdamState}
import dimwit.stats.{Normal, Uniform}

import io.circe.Json
import plotwit.*
import plotwit.PlotTargets.desktopBrowser

import deepwit.activation.gelu
import deepwit.base.{AffineFormLayer, AffineLayer}
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.loss.SquaredError

trait Sample derives Label
trait Batch derives Label
trait Feature derives Label
trait Embedding derives Label

class MLP(params: MLP.Params) extends (Tensor1[Feature, Float32] => Tensor0[Float32]):

  private val layer1 = AffineLayer(params.layer1)
  private val layer2 = AffineLayer(params.layer2)
  private val output = AffineFormLayer(params.output)

  def apply(x: Tensor1[Feature, Float32]): Tensor0[Float32] =
    val h1 = gelu(layer1(x))
    val h2 = gelu(layer2(h1))
    output(h2)

object MLP:

  case class Params(
      layer1: AffineLayer.Params[Feature, Embedding, Float32],
      layer2: AffineLayer.Params[Embedding, Embedding, Float32],
      output: AffineFormLayer.Params[Embedding, Float32]
  )

  object Params:

    def init(hiddenSize: Int, key: Key): Params =
      val (layer1Key, layer2Key, outputKey) = key.splitToTuple(3)
      val hidden1Extent = Axis[Embedding] -> hiddenSize
      val hidden2Extent = Axis[Embedding] -> hiddenSize
      Params(
        layer1 = AffineLayer.Params.init(NoisyCurve.featureExtent, hidden1Extent, layer1Key),
        layer2 = AffineLayer.Params.init(hidden1Extent, hidden2Extent, layer2Key),
        output = AffineFormLayer.Params.init(hidden2Extent, outputKey)
      )

def costFnFor(
    xs: Tensor2[Batch, Feature, Float32],
    ys: Tensor1[Batch, Float32]
)(params: MLP.Params): Tensor0[Float32] =
  val model = MLP(params)
  zipvmap(Axis[Batch])(xs, ys): (x, y) =>
    val yHat = model(x)
    SquaredError(y, yHat)
  .mean

case class TrainState(params: MLP.Params, optimizerState: AdamState[MLP.Params])

val checkpointPath = "out/Regression/checkpoint"

// -- Training --

@main
def train(): Unit =

  // -- Configuration --

  val numIterations = 3_000
  val numSamples = 512
  val batchSize = 64
  val hiddenSize = 32
  val learningRate = 3e-3f
  val noiseScale = 0.1f

  // -- Prepare training data --

  val (dataKey, initKey) = Key(42).split2()
  val trainBatchStream = NoisyCurve.sample(numSamples, noiseScale, dataKey).toBatchStream(Axis[Batch] -> batchSize)

  // -- Prepare train trajectory --

  val optimizer = Adam(learningRate)

  def gradientStep(batch: BatchSample, state: TrainState): TrainState =
    val grads = Autodiff.grad(costFnFor(batch.xs, batch.ys))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialState =
    val initialParams = MLP.Params.init(hiddenSize, initKey)
    TrainState(initialParams, optimizer.init(initialParams))

  // The trajectory is a lazy fold over the batches: state in, state out, one step per element.
  val trainTrajectory = trainBatchStream.scanLeft(initialState): (state, batch) =>
    jitGradientStep(batch, state)

  // -- Run train trajectory --

  val finalState = trainTrajectory
    .drop(numIterations)
    .next()

  // -- Save the fitted state --

  TensorTreeCheckpointer(checkpointPath, overwrite = true).save(finalState, numIterations)
  println(s"Done. Wrote $checkpointPath.")

// -- Evaluation --

@main
def eval(): Unit =

  val numSamples = 256
  val noiseScale = 0.1f
  val gridSize = 200

  val checkpointer = TensorTreeCheckpointer(checkpointPath)
  val state = checkpointer.load[TrainState](checkpointer.iterations.max).get
  val model = MLP(state.params)

  // A different key draws different points, so none of these were fitted to.
  val data = NoisyCurve.sample(numSamples, noiseScale, Key(7))
  val predictions = data.xs.vmap(Axis[Sample])(model)
  val cost = data.ys.zipvmap(Axis[Sample])(predictions)(SquaredError(_, _)).mean
  println(f"Held-out cost ${cost.item}%.4f, against the ${noiseScale * noiseScale}%.4f the noise alone puts there.")

  val grid = NoisyCurve.grid(gridSize)
  display(overlay(
    plots.scatterPlot(
      data.xs.slice(Axis[Feature].at(0)),
      data.ys,
      _.encoding.x.title := "x",
      _.encoding.y.title := "y",
      _.encoding.size := Json.obj("value" -> Json.fromInt(18))
    ),
    plots.linePlot(
      grid.slice(Axis[Feature].at(0)),
      grid.vmap(Axis[Sample])(model),
      _.title := "The model, over points it was never shown",
      _.encoding.color := Json.obj("value" -> Json.fromString("#f58518"))
    )
  ))

// -- Dataset --

case class BatchSample(xs: Tensor2[Batch, Feature, Float32], ys: Tensor1[Batch, Float32])

case class Dataset(xs: Tensor2[Sample, Feature, Float32], ys: Tensor1[Sample, Float32]):

  def toBatchStream(batchExtent: AxisExtent[Batch]): Iterator[BatchSample] =
    val count = xs.shape(Axis[Sample])
    Iterator.iterate(0)(_ + batchExtent.size).map: offset =>
      val ids = (0 until batchExtent.size).map(i => (offset + i) % count)
      BatchSample(
        xs.slice(Axis[Sample].at(ids)).relabel(Axis[Sample], Axis[Batch]),
        ys.slice(Axis[Sample].at(ids)).relabel(Axis[Sample], Axis[Batch])
      )

object NoisyCurve:

  val featureExtent: AxisExtent[Feature] = Axis[Feature] -> 1

  /** The interval the curve is defined over. */
  val (lower, upper) = (-1f, 1f)

  def truth[S: Label](x: Tensor1[S, Float32]): Tensor1[S, Float32] =
    (x *! (2f * math.Pi.toFloat)).sin + x *! 0.5f

  def sample(count: Int, noiseScale: Float, key: Key): Dataset =
    val (positionKey, noiseKey) = key.split2()
    val shape = Shape(Axis[Sample] -> count, featureExtent)
    val xs = Uniform(Tensor(shape).fill(lower), Tensor(shape).fill(upper)).sample(positionKey)
    val x = xs.slice(Axis[Feature].at(0))
    val noise = Normal.standardNormal(x.shape).sample(noiseKey) *! noiseScale
    Dataset(xs, truth(x) + noise)

  /** The interval, evenly sampled: the positions the model's curve is drawn through. */
  def grid(count: Int): Tensor2[Sample, Feature, Float32] =
    Tensor(Shape(Axis[Sample] -> count, featureExtent), VType[Float32])
      .fromArray(Array.tabulate(count)(i => lower + (upper - lower) * i / (count - 1)))
