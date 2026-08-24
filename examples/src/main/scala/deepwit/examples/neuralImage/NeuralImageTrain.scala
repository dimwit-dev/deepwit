package deepwit.examples.neuralImage

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.{Adam, AdamState}

import deepwit.training.{Monitor, tapEvery}
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.loss.SquaredError

case class TrainState(
    params: NeuralImage.Params,
    optimizerState: AdamState[NeuralImage.Params],
    lastCost: Tensor0[Float32]
)

trait Batch derives Label

case class TrainBatch(
    coordinates: Tensor2[Batch, PixelCoordinate, Float32],
    targets: Tensor2[Batch, Channel, Float32]
)

def costFnFor(
    trainBatch: TrainBatch
)(
    params: NeuralImage.Params
): Tensor0[Float32] =
  val model = NeuralImage(params)
  zipvmap(Axis[Batch])(trainBatch.coordinates, trainBatch.targets): (coordinate, target) =>
    val prediction = model(coordinate)
    target.zipvmap(Axis[Channel])(prediction)(SquaredError(_, _)).sum
  .mean

/** Stream of training batches. */
def batchStream(
    coordinates: Tensor2[Pixel, PixelCoordinate, Float32],
    targets: Tensor2[Pixel, Channel, Float32],
    batchSize: Int,
    key: Key
): Iterator[TrainBatch] =
  // permute across pixels to avoid biasing the network to any particular region of the image
  val permutation = Random.permutation(coordinates.shape.extent(Axis[Pixel]))(key)
  val coordinatesShuffled = coordinates.take(Axis[Pixel])(permutation)
  val targetsShuffled = targets.take(Axis[Pixel])(permutation)
  val totalPixels = coordinates.shape(Axis[Pixel])
  Iterator.iterate(0)(_ + batchSize).map: offset =>
    val batchIds = (0 until batchSize).map(i => (offset + i) % totalPixels)
    val batchPixels = coordinatesShuffled.slice(Axis[Pixel].at(batchIds)).relabel(Axis[Pixel], Axis[Batch])
    val batchLabels = targetsShuffled.slice(Axis[Pixel].at(batchIds)).relabel(Axis[Pixel], Axis[Batch])
    TrainBatch(batchPixels, batchLabels)

@main
def train(): Unit =

  // -- Configuration --

  val imageSize = 256
  val encodingSize = 64
  val hiddenSize = 256
  val batchSize = 4096
  val learningRate = 1e-3f
  val numIterations = 5_000

  // -- Prepare training data --

  val (initKey, dataKey) = Key(42).split2()

  val image = ImageLoader.load(imageSize)
  val pixels = image.flatten((Axis[Height], Axis[Width]))
  val coordinates = Coordinates.fourier(
    Coordinates.positions(image.shape.extent(Axis[Height])),
    Coordinates.positions(image.shape.extent(Axis[Width])),
    encodingSize
  )
  val trainBatchStream = batchStream(coordinates, pixels, batchSize, dataKey)

  // -- Prepare train trajectory --

  val optimizer = Adam(learningRate)

  def gradientStep(batch: TrainBatch, state: TrainState): TrainState =
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialState =
    val initialParams = NeuralImage.Params.init(
      coordinates.shape.extent(Axis[PixelCoordinate]),
      hiddenSize,
      pixels.shape.extent(Axis[Channel]),
      initKey
    )
    TrainState(initialParams, optimizer.init(initialParams), Tensor0(-1f))

  val trainTrajectory = trainBatchStream.scanLeft(initialState): (state, batch) =>
    jitGradientStep(batch, state)

  // -- Run train trajectory --

  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = _.lastCost.item)
  val finalState = trainTrajectory
    .tapEvery(100):
      case (state, step) => println(trainMonitor.report(step, state))
    .drop(numIterations)
    .next()

  // -- Save final state --

  val checkpointer = TensorTreeCheckpointer.newIn("out/NeuralImage")
  checkpointer.save(finalState, numIterations)
  println(f"Final cost: ${finalState.lastCost.item}%.6f")
  println(s"Done. Wrote ${checkpointer.rootPath}.")
