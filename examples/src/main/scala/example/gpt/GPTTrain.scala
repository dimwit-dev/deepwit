package example.gpt

import dimwit.*
import dimwit.jax.Jax
import dimwit.Conversions.given
import deepwit.*
import deepwit.labels.{Head, HeadQuery, HeadKey, HeadValue}
import nn.ActivationFunctions.gelu
import nn.Adam
import nn.AdamW
import dimwit.python.PyBridge.{toPyTensor, liftPyTensor, liftPyTensor1}
import dimwit.stats.Uniform
import dimwit.hardware.DeviceBackend.{CPU, GPU}
import me.shadaj.scalapy.py

object Config:
  val batchSize = 64
  val learningRate = 1e-4f
  val beta1 = 0.9f
  val beta2 = 0.99f
  val weightDecayFactor = 0.01f

  val numLayers = 12
  val vocabExtent = Axis[Vocab] -> 50257
  val contextExtent = Axis[Context] -> 1024
  val embeddingExtent = Axis[Embedding] -> 768
  val headExtent = Axis[Head] -> 12
  val headQueryExtent = Axis[HeadQuery] -> 64
  val headKeyExtent = Axis[HeadKey] -> 64
  val headValueExtent = Axis[HeadValue] -> 64
  val embeddingMixedExtent = Axis[MLPEmbeddingMixer.EmbeddingMixed] -> 3072

object DebugConfig:
  val batchSize = 1
  val learningRate = 1e-4f
  val beta1 = 0.9f
  val beta2 = 0.99f
  val weightDecayFactor = 0.01f

  val numLayers = 2
  val vocabExtent = Axis[Vocab] -> 50257
  val contextExtent = Axis[Context] -> 32
  val embeddingExtent = Axis[Embedding] -> 128
  val headExtent = Axis[Head] -> 4
  val headQueryExtent = Axis[HeadQuery] -> 32
  val headKeyExtent = Axis[HeadKey] -> 32
  val headValueExtent = Axis[HeadValue] -> 32
  val embeddingMixedExtent = Axis[MLPEmbeddingMixer.EmbeddingMixed] -> 512

import DebugConfig.*

case class BatchSample(
    targets: Tensor2[Sample, Context, Int],
    shiftedTargets: Tensor2[Sample, Context, Int]
)

@main def train(): Unit =

  val key = Random.Key.fromTime()

  val (dataKey, initParamsKey) = key.split2()

  val initParams = GPT.Params.init(numTransformerLayers = numLayers)(
    vocabExtent,
    contextExtent,
    headExtent,
    headQueryExtent,
    headKeyExtent,
    headValueExtent,
    embeddingExtent,
    embeddingMixedExtent,
    initParamsKey
  )

  val hyperParams = GPT.HyperParams(
    Transformer.HyperParams(
      TransformerLayer.HyperParams(MLPEmbeddingMixer.HyperParams(activationFunction = gelu)),
      LayerNorm.HyperParams(1e-12)
    )
  )

  val GPTModel = GPT(hyperParams)

  val adamW = AdamW(
    Adam(learningRate = learningRate, b1 = beta1, b2 = beta2),
    weightDecayFactor = weightDecayFactor
  )

  case class TrainingState(
      params: GPT.Params,
      adamWState: adamW.State[GPT.Params],
      stepCost: Tensor0[Float]
  )

  def loadData(binaryPath: String): Tensor1[Sample, Int] =
    lazy val np = py.module("numpy")
    liftPyTensor(Jax.jnp.asarray(
      np.memmap(binaryPath, dtype = np.uint16, mode = "r"),
      device = CPU.devices.head.toJaxDevice
    ))

  def loadBatch(data: Tensor1[Sample, Int], batchSize: Int, key: Random.Key): BatchSample =
    val maxIdx = data.shape(Axis[Sample]) - batchSize - 1
    val randomIndices = IndependentDistribution.fromUnivariate(
      Shape1(Axis[Sample] -> batchSize),
      Uniform(Tensor0(0), Tensor0(maxIdx))
    ).sample(key)
    val shiftedIndices = randomIndices +! 1
    val targets = randomIndices.vmap(Axis[Sample])(startIndex =>
      data.dynamicSlice(startIndex, contextExtent.size).relabelTo(Axis[Context])
    )
    val shiftedTargets = shiftedIndices.vmap(Axis[Sample])(startIndex =>
      data.dynamicSlice(startIndex, contextExtent.size).relabelTo(Axis[Context])
    )
    val gpu = GPU.devices.head
    BatchSample(targets.toDevice(gpu), shiftedTargets.toDevice(gpu))

  def batchStream(data: Tensor1[Sample, Int], batchSize: Int, initialKey: Random.Key): LazyList[BatchSample] =
    val stateStream = LazyList.iterate((loadBatch(data, batchSize, initialKey), initialKey)):
      case (_, prevKey) =>
        val (nowKey, nextKey) = prevKey.split2()
        (loadBatch(data, batchSize, nowKey), nextKey)
    stateStream.map(_._1)

  val (trainKey, valKey) = dataKey.split2()
  val trainStream = batchStream(loadData("data/openwebtext/val.bin"), batchSize, trainKey)

  def loss(
      targets: Tensor1[Context, Int],
      logits: Tensor2[Context, Vocab, Float]
  ): Tensor0[Float] =
    zipvmap(Axis[Context])(targets, logits)(CategoricalCrossEntropy.fromLogits).mean

  def costFunFor(
      batchSample: BatchSample
  )(
      params: GPT.Params
  ): Tensor0[Float] =
    val model = GPTModel(params)
    val losses = zipvmap(Axis[Sample])(batchSample.targets, batchSample.shiftedTargets):
      case (targets, shiftedTargets) =>
        val logits = model.logits(shiftedTargets)
        loss(targets, logits)
    losses.mean

  def gradientStep(
      batchSample: BatchSample,
      state: TrainingState
  ): TrainingState =
    val costFn = costFunFor(batchSample)
    val grads = Autodiff.grad(costFn)(state.params)
    val stepCost = costFn(state.params) // TODO move to gradAndValue
    val (params, adamWState) = adamW.update(grads, state.params, state.adamWState)
    TrainingState(params, adamWState, stepCost)

  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  def miniBatchGradientDescent(
      samples: LazyList[BatchSample],
      startState: TrainingState
  ): LazyList[TrainingState] =
    samples.scanLeft(startState):
      case (state, sample) =>
        dimwit.gc()
        jitGradientStep(sample, state)

  val initState = TrainingState(initParams, adamW.init(initParams), Tensor0(-1f))
  val trainTrajectory = miniBatchGradientDescent(trainStream, initState)
  val timer = Timer.start()
  println("Training...")
  val finalState = trainTrajectory
    .drop(1)
    .tapEvery(1):
      case (state, iter) =>
        // Training report
        timer.tick()
        val secondsPerBatch = timer.runningAvgSeconds
        println(
          List(
            s"iter $iter",
            f"samples/s: ${batchSize / (secondsPerBatch)}%.2f",
            f"s/batch: $secondsPerBatch%.2f",
            f"stepCost ${state.stepCost.item}%.2f"
          ).mkString(", ")
        )
    .drop(1_000_000_000)
    .head
