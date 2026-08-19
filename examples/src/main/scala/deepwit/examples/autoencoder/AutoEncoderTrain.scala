package deepwit.examples.autoencoder

import dimwit.*
import dimwit.Conversions.given

import dimwit.optimizer.GradientDescent
import dimwit.optimizer.GradientDescentState

import deepwit.examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import deepwit.training.{Monitor, tapEvery}
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.loss.BinaryCrossEntropy

trait Batch derives Label

case class TrainState(params: Autoencoder.Params, optimizerState: GradientDescentState[Autoencoder.Params], lastCost: Tensor0[Float32])

@main
def autoEncoderTraining(): Unit =

  val batchSize = 512
  val numIterations = 5000
  val latentDim = 20
  val learningRate = 1e-3f

  val eHidden1Extent = Axis[EHidden1] -> 512
  val eHidden2Extent = Axis[EHidden2] -> 256
  val latentExtent = Axis[Latent] -> latentDim
  val dHidden1Extent = Axis[DHidden1] -> 256
  val dHidden2Extent = Axis[DHidden2] -> 512

  val initKey = Random.Key.fromTime()

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val testDataset = MNISTLoader.createTestDataset().get

  val trainDataStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize).map(_.images)

  def costFnFor[S: Label](samples: Tensor3[S, Height, Width, Float32])(params: Autoencoder.Params): Tensor0[Float32] =
    val model = Autoencoder(params)
    samples
      .vmap(Axis[S]): sample =>
        val original = sample.flatten
        zipvmap(Axis[Pixel])(original, model.logits(original)): (origPixel, reconPixel) =>
          BinaryCrossEntropy.fromLogits(origPixel, reconPixel)
        .sum
      .mean

  val optimizer = GradientDescent(learningRate = learningRate)

  def gradientStep(batch: Tensor3[Batch, Height, Width, Float32], state: TrainState): TrainState =
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialParams = Autoencoder.Params.init(
    eHidden1Extent,
    eHidden2Extent,
    latentExtent,
    dHidden1Extent,
    dHidden2Extent,
    initKey
  )
  val initialOptimizerState = optimizer.init(initialParams)
  val trainTrajectory = trainDataStream.scanLeft(TrainState(initialParams, initialOptimizerState, Tensor0(-1f))):
    case (state, batch) =>
      dimwit.gc()
      jitGradientStep(batch, state)

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val logger = new TensorTreeCheckpointer(f"out/AutoEncoder/$time")

  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)

  val state = trainTrajectory
    .tapEvery(10):
      case (state, step) =>
        println(trainMonitor.report(step, state))
    .tapEvery(500):
      case (state, step) =>
        val lossValue = costFnFor(testDataset.images)(state.params).item
        println(s"Step $step | Test loss: $lossValue")
    .tapEvery(500):
      case (state, step) =>
        logger.save(state, step)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .next()

  println("Done")
