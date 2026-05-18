/** My attempt to add gradient accumulation => Leads to OOM...
  */

package example.gpt

object BACKUP:

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
  import dimwit.FloatTree.ops.*

  import java.io.{FileWriter, PrintWriter, File}
  import java.time.LocalDateTime
  import java.time.format.DateTimeFormatter

  import dimwit.FloatTree.map
  import me.shadaj.scalapy.py

  object Config:
    val runningBatchSize = 48 // 12
    val effectiveBatchSize = 480
    val accumulationSteps = effectiveBatchSize / runningBatchSize
    val learningRate = 6e-4f
    val beta1 = 0.9f
    val beta2 = 0.95f
    val gradientClipNorm: Float = 1.0f
    val weightDecayFactor = 0.1f

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

  import Config.*

  case class BatchSample(
      targets: Tensor2[Sample, Context, Int32],
      inputs: Tensor2[Sample, Context, Int32]
  )

  @main def train2(): Unit =

    /** Helper-Type to mark np.memmap tensor */
    type LazyTensor1[L, V] = Tensor1[L, V]

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
      VType[Float32],
      initParamsKey
    )

    val hyperParams = GPT.HyperParams(
      Transformer.HyperParams(
        LayerNorm.HyperParams(1e-12)
      )
    )

    val adamW = AdamW(
      Adam(learningRate = learningRate, b1 = beta1, b2 = beta2),
      weightDecayFactor = weightDecayFactor
    )

    case class TrainingState(
        params: GPT.Params[Float32],
        adamWState: adamW.State[GPT.Params[Float32]],
        stepCost: Tensor0[Float32]
    )

    def loadData(binaryPath: String): LazyTensor1[Sample, UInt16] =
      lazy val np = py.module("numpy")
      liftPyTensor(
        np.memmap(binaryPath, dtype = np.uint16, mode = "r")
      )

    def loadBatch(data: LazyTensor1[Sample, UInt16], batchSize: Int, key: Random.Key): BatchSample =
      val maxIdx = data.shape(Axis[Sample]) - batchSize - 1
      val randomIndices = IndependentDistribution.fromUnivariate(
        Shape1(Axis[Sample] -> batchSize),
        Uniform(Tensor0(0), Tensor0(maxIdx))
      ).sample(key)
      val shiftedIndices = randomIndices +! 1
      val inputs = randomIndices.vmap(Axis[Sample])(startIndex =>
        data.dynamicSlice(startIndex, contextExtent.size).relabelTo(Axis[Context])
      )
      val targets = shiftedIndices.vmap(Axis[Sample])(startIndex =>
        data.dynamicSlice(startIndex, contextExtent.size).relabelTo(Axis[Context])
      )
      val gpu = GPU.devices.head
      BatchSample(targets.asInt32.toDevice(gpu), inputs.asInt32.toDevice(gpu))

    def batchStream(data: Tensor1[Sample, UInt16], batchSize: Int, initialKey: Random.Key): Iterator[BatchSample] =
      val stateStream = Iterator.iterate((loadBatch(data, batchSize, initialKey), initialKey)):
        case (_, prevKey) =>
          val (nowKey, nextKey) = prevKey.split2()
          (loadBatch(data, batchSize, nowKey), nextKey)
      stateStream.map(_._1)

    val (trainKey, valKey) = dataKey.split2()
    val trainStream = batchStream(loadData("data/openwebtext/train.bin"), runningBatchSize, trainKey)

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
      val model = GPT(hyperParams)(params)
      val losses = zipvmap(Axis[Sample])(batchSample.targets, batchSample.inputs):
        case (targets, inputs) =>
          val logits = model.logits(inputs)
          loss(targets, logits)
      losses.mean

    def calcGradients(
        batchSample: BatchSample,
        params: GPT.Params[BFloat16]
    ): (Tensor0[BFloat16], Grad[GPT.Params[BFloat16]]) =
      val costFn = costFunFor[BFloat16](batchSample)
      Autodiff.valueAndGrad(costFn)(params)

    val jitCalcGradients = jit(calcGradients)
    val jitAdamWUpdate = jitDonatingUnsafe(adamW.update[GPT.Params[Float32]])

    def gradientDescentStep(
        runningBatchSamples: List[BatchSample],
        state: TrainingState
    ): TrainingState =
      val paramsF16 = state.params.asFloats(VType[BFloat16])

      val initialGrads = state.params.map([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, Float32]) => Tensor.like(x).fill(0f))
      val (accumulatedCosts, accumulatedGrads) =
        runningBatchSamples.foldLeft((Tensor0(0f), initialGrads)):
          case ((accCosts, accGrads), batchSample) =>
            val (costs, grads) = jitCalcGradients(batchSample, paramsF16)

            val newAccCosts = accCosts + costs.asFloat32 / accumulationSteps
            val newAccGrads = accGrads ++ grads.value.asFloats(VType[Float32]) `//!` accumulationSteps

            // Hypothis: The return values from a jitted function seem to be kept alive, maybe by ScalaPy?
            dimwit.python.PyBridge.toPyTensor(costs).addressable_data(0).delete()
            grads.value.map([T <: Tuple] =>
              (labels: Labels[T]) ?=>
                (x: Tensor[T, BFloat16]) =>
                  dimwit.python.PyBridge.toPyTensor(x).addressable_data(0).delete()
                  x
            )

            // Hypothis: foldLeft accumulation variables seem to be kept alive. No idea why.
            dimwit.python.PyBridge.toPyTensor(accCosts).addressable_data(0).delete()
            accGrads.map([T <: Tuple] =>
              (labels: Labels[T]) ?=>
                (x: Tensor[T, Float32]) =>
                  dimwit.python.PyBridge.toPyTensor(x).addressable_data(0).delete()
                  x
            )

            (newAccCosts, newAccGrads)

      val (params, adamWState) = jitAdamWUpdate(Grad(accumulatedGrads), state.params, state.adamWState)

      val scalarLoss = accumulatedCosts.item

      // crashes after 50 iterations without this! WTF
      dimwit.python.PyBridge.toPyTensor(accumulatedCosts).addressable_data(0).delete()
      accumulatedGrads.map([T <: Tuple] =>
        (labels: Labels[T]) ?=>
          (x: Tensor[T, Float32]) =>
            dimwit.python.PyBridge.toPyTensor(x).addressable_data(0).delete()
            x
      )

      TrainingState(params, adamWState, Tensor0(scalarLoss))

    val jitGradientDescentStep = dimwit.eagerCleanup(gradientDescentStep)
    // val jitGradientDescentStep = gradientDescentStep
    val initState = TrainingState(initParams, adamW.init(initParams), Tensor0(-1f))

    def miniBatchGradientDescent(
        samples: Iterator[BatchSample],
        startState: TrainingState
    ): Iterator[TrainingState] =
      samples.grouped(accumulationSteps).scanLeft(startState):
        case (state, runningBatches) =>
          dimwit.gc()
          jitGradientDescentStep(runningBatches.toList, state)

    val trainTrajectory = miniBatchGradientDescent(trainStream, initState)

    // Initialize CSV File
    val csvFile = new File(s"training_log_${LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))}.csv")
    val writer = new PrintWriter(new FileWriter(csvFile, true), true)

    val headers = List("timestamp", "iter", "samples_per_s", "s_per_batch", "step_cost")
    writer.println(headers.mkString(","))

    val timer = Timer.start()
    println("Training...")
    val finalState = trainTrajectory
      .drop(1)
      .tapEvery(1):
        case (state, iter) =>
          // Training report
          timer.tick()
          val secondsPerBatch = timer.runningAvgSeconds
          val logData = Map(
            "timestamp" -> java.time.Instant.now().toString,
            "iter" -> iter,
            "samples_per_s" -> f"${effectiveBatchSize / (secondsPerBatch)}%.2f",
            "s_per_batch" -> f"$secondsPerBatch%.2f",
            "step_cost" -> f"${state.stepCost.item}%.2f"
          )
          writer.println(headers.map(h => logData(h)).mkString(","))
          println(headers.map(h => s"$h: ${logData(h)}").mkString(" | "))
      .drop(1_000_000_000)
      .next()
