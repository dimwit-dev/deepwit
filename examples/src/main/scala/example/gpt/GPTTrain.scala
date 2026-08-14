/** TODO
  * - GradScaler and BFloat16 support
  * - Increase batch size (gradient accumulation)
  */

package example.gpt

import dimwit.*
import dimwit.jax.Jax
import dimwit.Conversions.given
import deepwit.*
import deepwit.transformer.attention.{Head, HeadQuery, HeadKey, HeadValue}
import nn.ActivationFunctions.gelu
import dimwit.optimizer.{Adam, AdamState, AdamW}
import dimwit.python.PyBridge.{toPyTensor, liftPyTensor, liftPyTensor1}
import dimwit.stats.Uniform
import dimwit.hardware.DeviceBackend.{CPU, GPU}
import dimwit.TreeOf.ops.asFloats
import me.shadaj.scalapy.py
import FineWebDataset.{BatchSample, batchStream}
import deepwit.transformer.MLPEmbeddingMixer
import deepwit.loss.CategoricalCrossEntropy

object Config:
  val batchSize = 64
  val learningRate = 1e-4f
  val beta1 = 0.9f
  val beta2 = 0.99f
  val weightDecayFactor = 0.01f

  val numLayers = 12
  val vocabExtent = Axis[Vocab] -> 50304
  val contextExtent = Axis[Context] -> 1024
  val numHeads = 12
  val embeddingExtent = Axis[Embedding] -> 64 * numHeads
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
  val numHeads = 4
  val embeddingExtent = Axis[Embedding] -> 32 * numHeads
  val embeddingMixedExtent = Axis[MLPEmbeddingMixer.EmbeddingMixed] -> 512

import Config.*

@main def train(): Unit =

  /** Helper-Type to mark np.memmap tensor */

  val key = Random.Key.fromTime()

  val (dataKey, initParamsKey) = key.split2()

  val initParams = GPT.Params.init(numTransformerLayers = numLayers)(
    vocabExtent,
    contextExtent,
    numHeads,
    embeddingExtent,
    embeddingMixedExtent,
    VType[Float32],
    initParamsKey
  )

  val adamW = AdamW(
    Adam(learningRate = learningRate, beta1 = beta1, beta2 = beta2),
    weightDecayFactor = weightDecayFactor
  )

  case class TrainingState(
      params: GPT.Params[Float32],
      adamWState: AdamState[GPT.Params[Float32]],
      stepCost: Tensor0[Float32]
  )

  val (trainKey, valKey) = dataKey.split2()
  val trainStream = batchStream("/home/mebr/Documents/Scala/modded-nanogpt/data/fineweb10B", "fineweb_train_", batchSize, contextExtent.size, trainKey)

  def loss[V: IsFloating](
      targets: Tensor1[Context, Int32],
      logits: Tensor2[Context, Vocab, V]
  ): Tensor0[V] =
    zipvmap(Axis[Context])(targets, logits)(CategoricalCrossEntropy.fromLogits).mean

  def costFunFor[V: IsFloating](
      batchSample: BatchSample
  )(
      params: GPT.Params[V]
  ): Tensor0[V] =
    val model = GPT(params)
    val losses = zipvmap(Axis[Sample])(batchSample.targets, batchSample.inputs):
      case (targets, inputs) =>
        val logits = model.logits(inputs)
        loss(targets, logits)
    losses.mean

  def gradientStep[V: IsFloating](
      batchSample: BatchSample,
      state: TrainingState
  ): TrainingState =
    val costFn = costFunFor[BFloat16](batchSample)
    val paramsF16 = state.params.asFloats(VType[BFloat16])
    val (stepCost, grads) = Autodiff.valueAndGrad(costFn)(paramsF16)
    val gradsF32 = Grad.asFloats(grads)(VType[Float32])
    val (params, adamWState) = adamW.update(gradsF32, state.params, state.adamWState)
    TrainingState(params, adamWState, stepCost.asFloat32)

  val jitGradientStep = jitDonatingUnsafe(gradientStep[Float32])

  def miniBatchGradientDescent(
      samples: Iterator[BatchSample],
      startState: TrainingState
  ): Iterator[TrainingState] =
    samples.scanLeft(startState):
      case (state, sample) =>
        System.gc()
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
            f"tokens/s: ${(batchSize * contextExtent.size) / (secondsPerBatch)}%.2f",
            f"samples/s: ${batchSize / (secondsPerBatch)}%.2f",
            f"s/batch: $secondsPerBatch%.2f",
            f"stepCost ${state.stepCost.item}%.2f"
          ).mkString(", ")
        )
    .drop(1_000_000_000)
    .next()
