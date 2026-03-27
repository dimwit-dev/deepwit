package example.autoencoder

import examples.timed
import dimwit.*
import dimwit.Conversions.given
import deepwit.*

import dimwit.stats.Normal
import dimwit.random.Random
import nn.ActivationFunctions.relu
import nn.GradientDescent
import dimwit.jax.Jax
import nn.ActivationFunctions.sigmoid
import dimwit.random.Random.Key

import examples.dataset.MNISTLoader
import MNISTLoader.{Sample, TrainSample, TestSample}
import dimwit.python.PyBridge.toPyTensor
import deepwit.logging.TenZarrLogger

private trait Batch derives Label

case class TrainState(params: Autoencoder.Params, lastCost: Tensor0[Float])

@main
def autoEncoderTraining(): Unit =

  val learningRate = 1e-3f

  val batchSize = 512
  val numIterations = 5000
  val latentDim = 20

  val eHidden1Extent = Axis[EHidden1] -> 512
  val eHidden2Extent = Axis[EHidden2] -> 256
  val latentExtent = Axis[Latent] -> latentDim
  val dHidden1Extent = Axis[DHidden1] -> 256
  val dHidden2Extent = Axis[DHidden2] -> 512

  val initKey = Random.Key.fromTime()

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val testDataset = MNISTLoader.createTestDataset().get

  val trainDataStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize).map(_.images)

  def costFnFor[S: Label](samples: Tensor3[S, Height, Width, Float])(params: Autoencoder.Params): Tensor0[Float] =
    val model = Autoencoder(params)
    samples
      .vmap(Axis[S]): sample =>
        val original = sample.flatten
        zipvmap(Axis[Pixel])(original, model.logits(original)): (origPixel, reconPixel) =>
          BinaryCrossEntropy.fromLogits(origPixel, reconPixel)
        .sum
      .mean

  val optimizer = GradientDescent(learningRate = Tensor0(learningRate))

  def gradientStep(batch: Tensor3[Batch, Height, Width, Float], state: TrainState): TrainState =
    val grads = Autodiff.grad(costFnFor(batch))(state.params)
    val cost = costFnFor(batch)(state.params)
    val (newParams, _) = optimizer.update(grads, state.params, ())
    TrainState(newParams, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialParams = Autoencoder.Params.xavierNormal(
    eHidden1Extent,
    eHidden2Extent,
    latentExtent,
    dHidden1Extent,
    dHidden2Extent,
    initKey
  )
  val trainTrajectory = trainDataStream.scanLeft(TrainState(initialParams, Tensor0(-1f))):
    case (state, batch) =>
      dimwit.gc()
      jitGradientStep(batch, state)

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val logger = new TenZarrLogger(f"out/AutoEncoder/$time")

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
        logger.logTensorTree("checkpoint", step, state)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .head

  println("Done")
