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

trait Batch derives Label

def binaryCrossEntropy[L: Label](
    target: Tensor1[L, Float],
    pred: Tensor1[L, Float],
    eps: Float = 1e-7f
): Tensor0[Float] =
  val p = pred.clip(eps, 1f - eps) // Clamp predictions to [eps, 1 - eps] for numerical stability
  val loss = -((target * p.log) + ((Tensor0(1f) -! target) * (1f -! p).log))
  loss.sum

case class TrainState(params: Autoencoder.Params, lastCost: Tensor0[Float])

@main
def autoEncoderTraining(): Unit =

  val learningRate = 1e-3f

  val batchSize = 512
  val numIterations = 5000
  val latentDim = 20

  val eHidden1Extent = Axis[EHidden1] -> 512
  val eHidden2Extent = Axis[EHidden2] -> 256
  val latentExtent = Axis[Latent] -> 20
  val dHidden1Extent = Axis[DHidden1] -> 256
  val dHidden2Extent = Axis[DHidden2] -> 512

  val initKey = Random.Key.fromTime()

  val (trainX, trainY) = MNISTLoader.createTrainingDataset().get
  val (testX, testY) = MNISTLoader.createTestDataset().get

  def costFnFor[S: Label](samples: Tensor3[S, Height, Width, Float])(params: Autoencoder.Params): Tensor0[Float] =
    val model = Autoencoder(params)
    samples
      .vmap(Axis[S]): sample =>
        val original = sample.flatten
        binaryCrossEntropy(original, model(original))
      .mean

  def batchStream[S: Label](
      imgs: Tensor3[S, Height, Width, Float],
      labels: Tensor1[S, Int],
      batchSize: Int
  ): LazyList[Tensor3[Batch, Height, Width, Float]] =
    val totalSamples = imgs.shape(Axis[S])
    LazyList.iterate(0)(_ + batchSize).map: offset =>
      val batchIds = (0 until batchSize).map(i => (offset + i) % totalSamples)
      imgs.slice(Axis[S].at(batchIds)).relabel(Axis[S], Axis[Batch])

  val trainDataStream = batchStream(trainX, trainY, batchSize)

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
      case (state, epoch) =>
        val lossValue = costFnFor(testX)(state.params).item
        println(s"Epoch $epoch | Test loss: $lossValue")
    .tapEvery(500):
      case (state, epoch) =>
        println("Saving checkpoint...")
        logger.logTensorTree("checkpoint", epoch, state)
    .drop(numIterations)
    .head

  println("Done")
