package deepwit.example.diffusion

import dimwit.*
import dimwit.Conversions.given

import deepwit.*

import examples.timed
import examples.dataset.{MNISTLoader, MNISTBatchSample}
import dimwit.stats.Normal
import dimwit.stats.Uniform

import MNISTLoader.{Width, Height}
import dimwit.optimizer.{AdamW, Adam, AdamState}

import dimwit.tensortree.TreeOf

private trait Batch derives Label

case class TrainState(params: DiffusionUNet.Params, state: AdamState[DiffusionUNet.Params], lastCost: Tensor0[Float32], key: Random.Key) derives TensorTree

@main
def train(): Unit =

  val learningRate = 3e-4f
  val numIterations = 10_000
  val batchSize = 32

  val trainKey = Random.Key(42)
  val (initKey, loopKey) = trainKey.splitToTuple(2)

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val trainDataBatchStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize)

  val hyperParams = DiffusionUNet.HyperParams.default

  def costFnFor[S: Label](
      images: Tensor3[S, Height, Width, Float32],
      keys: Tensor1[S, Random.Key]
  )(params: DiffusionUNet.Params): Tensor0[Float32] =
    val model = DiffusionUNet(hyperParams)(params)
    zipvmap(Axis[S])(images, keys): (img, key) =>
      val (timeKey, noiseKey) = key.item.splitToTuple(2)

      val u = Uniform(Tensor0(0f), Tensor0(1f)).sample(timeKey)
      val t = 1f - u.pow(3) // skew towards bigger timesteps for more challenging denoising tasks
      val alpha = (1f - t).sqrt
      val sigma = t.sqrt

      val scaledImg = img *! 2.0f -! 1.0f // Scale to [-1, 1] for better training stability
      val stdNoise = Normal.standardNormal(scaledImg.shape).sample(noiseKey)
      val noise = stdNoise *! sigma
      val noisyImage = scaledImg *! alpha + noise
      val predictedNoise = model(noisyImage, t)

      (predictedNoise - stdNoise).pow(2).mean
    .mean

  val optimizer = AdamW(Adam(learningRate = learningRate), weightDecayFactor = 0.01f)

  def gradientStep(
      batch: MNISTBatchSample[Batch],
      trainState: TrainState
  ): TrainState =
    val (stepKey, nextKey) = trainState.key.splitToTuple(2)
    val keys = stepKey.splitToTensor(Axis[Batch] -> batchSize)

    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch.images, keys))(trainState.params)
    val (newParams, state) = optimizer.update(grads, trainState.params, trainState.state)

    TrainState(newParams, state, cost, nextKey)

  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialParams = DiffusionUNet.Params.xavierUniform(List(32, 64, 128))(initKey)
  val initialOptimizerState = optimizer.init(initialParams)
  val initialTrainState = TrainState(initialParams, initialOptimizerState, Tensor0(-1f), loopKey)

  val trainTrajectory = trainDataBatchStream.scanLeft(initialTrainState):
    case (trainState, batch) =>
      dimwit.gc()
      jitGradientStep(batch, trainState)

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val logger = new TensorTreeLogger(f"out/DiffusionMNIST/$time")
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)

  val state = trainTrajectory
    .tapEvery(10):
      case (state, step) => println(trainMonitor.report(step, state))
    .tapEvery(500):
      case (state, step) =>
        logger.save(state, step)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .next()
