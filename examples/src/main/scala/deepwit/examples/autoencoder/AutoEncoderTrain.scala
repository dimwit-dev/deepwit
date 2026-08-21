package deepwit.examples.autoencoder

import dimwit.*
import dimwit.Conversions.given

import deepwit.examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import deepwit.training.{Monitor, tapEvery}
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.loss.BinaryCrossEntropy
import dimwit.optimizer.{Adam, AdamState}

case class TrainState(params: Autoencoder.Params, optimizerState: AdamState[Autoencoder.Params], lastCost: Tensor0[Float32])

@main
def train(): Unit =
  // -- Configuration --

  val batchSize = 256
  val numIterations = 3_000
  val latentDim = 24
  val learningRate = 3e-4f

  val eHidden1Extent = Axis[EHidden1] -> 512
  val eHidden2Extent = Axis[EHidden2] -> 256
  val latentExtent = Axis[Latent] -> latentDim
  val dHidden1Extent = Axis[DHidden1] -> 256
  val dHidden2Extent = Axis[DHidden2] -> 512

  // -- Prepare train trajectory --
  trait Batch derives Label

  val initKey = Key(42)

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val testDataset = MNISTLoader.createTestDataset().get

  val trainDataStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize).map(_.images)
  val optimizer = Adam(learningRate = learningRate)

  def costFnFor[S: Label](samples: Tensor3[S, Height, Width, Float32])(params: Autoencoder.Params): Tensor0[Float32] =
    val model = Autoencoder(params)
    samples
      .vmap(Axis[S]): sample =>
        val original = sample.flatten
        zipvmap(Axis[Pixel])(original, model.logits(original)): (origPixel, reconPixel) =>
          BinaryCrossEntropy.fromLogits(origPixel, reconPixel)
        .sum
      .mean

  def gradientStep(batch: Tensor3[Batch, Height, Width, Float32], state: TrainState): TrainState =
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialState =
    val initialParams = Autoencoder.Params.init(eHidden1Extent, eHidden2Extent, latentExtent, dHidden1Extent, dHidden2Extent, initKey)
    val initialOptimizerState = optimizer.init(initialParams)
    TrainState(initialParams, initialOptimizerState, Tensor0(-1f))

  val trainTrajectory = trainDataStream.scanLeft(initialState):
    case (state, batch) =>
      jitGradientStep(batch, state)

  // -- Run train trajectory --

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val checkpointPath = f"out/AutoEncoder/$time"
  val checkpointer = new TensorTreeCheckpointer(checkpointPath)
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)
  val finalState = trainTrajectory
    .tapEvery(10):
      // Report training loss
      case (state, step) =>
        println(trainMonitor.report(step, state))
    .tapEvery(500):
      // Evaluate on the test set
      case (state, step) =>
        val lossValue = costFnFor(testDataset.images)(state.params).item
        println(s"Step $step | Test loss: $lossValue")
    .tapEvery(500):
      // Save checkpoints
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .next()

  println(s"Done. Wrote $checkpointPath.")
