/** My attempt to add gradient accumulation => Leads to OOM... */

package deepwit.examples.gpt
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.transformer.EmbeddingMixed
import deepwit.loss.CategoricalCrossEntropy

object BACKUP:

  import dimwit.*
  import dimwit.Conversions.given
  import deepwit.training.tapEvery
  import deepwit.optimizer.*
  import dimwit.optimizer.{AdamW, Adam, AdamState}
  import dimwit.TreeOf.ops.*

  import java.io.{FileWriter, PrintWriter, File}
  import java.time.LocalDateTime
  import java.time.format.DateTimeFormatter
  import FineWebDataset.{BatchSample, batchStream}

  import dimwit.TreeOf.map
  import me.shadaj.scalapy.py

  import deepwit.optimizer.LearningRateSchedule
  import deepwit.optimizer.LearningRateSchedules.LinearWarmup

  object Config:
    val runningBatchSize = 64 // 12
    val effectiveBatchSize = 512
    val accumulationSteps = effectiveBatchSize / runningBatchSize
    val baseLearningRate = 6e-4f
    val minLearningRate = baseLearningRate / 10f
    val beta1 = 0.9f
    val beta2 = 0.95f
    val gradientClipNorm: Float = 1.0f
    val weightDecayFactor = 0.1f

    val numLayers = 12
    val vocabExtent = Axis[Vocab] -> 50304
    val contextExtent = Axis[Context] -> 1024
    val numHeads = 12
    val embeddingExtent = Axis[Embedding] -> 64 * numHeads
    val embeddingMixedExtent = Axis[EmbeddingMixed] -> 3072

    val valTokens = 10485760
    val valSamples = valTokens / contextExtent.size
    val valRunningBatchSize = runningBatchSize
    val numBatchesPerValidation = valSamples / valRunningBatchSize

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
    val embeddingMixedExtent = Axis[EmbeddingMixed] -> 512

  import Config.*

  @main def train2(): Unit =

    /** Helper-Type to mark np.memmap tensor */
    type LazyTensor1[L, V] = Tensor1[L, V]

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

    val schedule = LinearWarmup(baseLearningRate, 1_000).followBy(CosineDecay(baseLearningRate, minLearningRate, 20_000))
    val opt = LearningRateScheduler(lr => AdamW(Adam(lr, beta1 = beta1, beta2 = beta2), weightDecayFactor = weightDecayFactor), schedule)

    case class TrainingState(
        params: GPT.Params[Float32],
        optState: LearningRateSchedulerState[GPT.Params[Float32], AdamState],
        stepCost: Tensor0[Float32]
    )

    val (trainKey, valKey) = dataKey.split2()
    val trainStream = batchStream("/home/mebr/Documents/Scala/modded-nanogpt/data/fineweb10B", "fineweb_train_", runningBatchSize, contextExtent.size, trainKey)
    val valStream = batchStream("/home/mebr/Documents/Scala/modded-nanogpt/data/fineweb10B", "fineweb_val_", runningBatchSize, contextExtent.size, valKey)

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
    val jitCostFn = jit: (params: GPT.Params[BFloat16], batchSample: BatchSample) =>
      costFunFor(batchSample)(params)

    def calcGradients(
        batchSample: BatchSample,
        params: GPT.Params[BFloat16]
    ): (Tensor0[BFloat16], Grad[GPT.Params[BFloat16]]) =
      val costFn = costFunFor[BFloat16](batchSample)
      Autodiff.valueAndGrad(costFn)(params)

    val jitCalcGradients = jit(calcGradients)
    val jitAdamWUpdate = jitDonatingUnsafe(opt.update[GPT.Params[Float32], Float32])

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

      val (params, optState) = jitAdamWUpdate(
        Grad(accumulatedGrads).clipGlobalNorm(gradientClipNorm),
        state.params,
        state.optState
      )

      val scalarLoss = accumulatedCosts.item

      // crashes after 50 iterations without this! WTF
      dimwit.python.PyBridge.toPyTensor(accumulatedCosts).addressable_data(0).delete()
      accumulatedGrads.map([T <: Tuple] =>
        (labels: Labels[T]) ?=>
          (x: Tensor[T, Float32]) =>
            dimwit.python.PyBridge.toPyTensor(x).addressable_data(0).delete()
            x
      )

      TrainingState(params, optState, Tensor0(scalarLoss))

    // val jitGradientDescentStep = dimwit.eagerCleanup(gradientDescentStep)
    val jitGradientDescentStep = gradientDescentStep

    val initState = TrainingState(initParams, opt.init(initParams), Tensor0(-1f))
    // println("Load checkpoint!")
    // val initState = new TenZarrLogger(f"out/GPT-2/20260526_065320").loadTensorTree[TrainingState](initState1, "checkpoint", 18_006).get

    def miniBatchGradientDescent(
        samples: Iterator[BatchSample],
        startState: TrainingState
    ): Iterator[TrainingState] =
      samples.grouped(accumulationSteps).scanLeft(startState):
        case (state, runningBatches) =>
          // dimwit.gc()
          System.gc() // TODO performance test this as Jax.gc() is stop-the-world
          jitGradientDescentStep(runningBatches.toList, state)

    val trainTrajectory = miniBatchGradientDescent(trainStream, initState)

    // Initialize CSV File
    val time = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
    val logger = TensorTreeCheckpointer(f"out/GPT-2/$time")

    val csvFile = new File(s"training_log_$time.csv")
    val writer = new PrintWriter(new FileWriter(csvFile, true), true)

    val headers = List("timestamp", "iter", "tokens_per_s", "samples_per_s", "s_per_batch", "learning_rate", "step_cost")
    writer.println(headers.mkString(","))

    val timer = Timer.start()
    println("Training...")

    val finalState = trainTrajectory
      .drop(1)
      .tapEvery(1):
        case (state, _) =>
          val step = state.optState.step.item
          // Training report
          timer.tick()
          val secondsPerBatch = timer.runningAvgSeconds
          val logData = Map(
            "timestamp" -> java.time.Instant.now().toString,
            "step" -> step,
            "tokens_per_s" -> f"${(effectiveBatchSize * contextExtent.size) / (secondsPerBatch)}%.2f",
            "samples_per_s" -> f"${effectiveBatchSize / (secondsPerBatch)}%.2f",
            "s_per_batch" -> f"$secondsPerBatch%.2f",
            "learning_rate" -> f"${schedule(step)}",
            "step_cost" -> f"${state.stepCost.item}%.2f"
          )
          writer.println(headers.map(h => logData(h)).mkString(","))
          println(headers.map(h => s"$h: ${logData(h)}").mkString(" | "))
      .tapEvery(1_000):
        case (state, _) =>
          val step = state.optState.step.item
          println("-" * 30)
          println(s"Performing validation at step $step...")
          val params = state.params.asFloats(VType[BFloat16])
          val avgValLoss = (1 to numBatchesPerValidation).iterator
            .map:
              case _ =>
                val valBatch = valStream.next()
                jitCostFn(params, valBatch).asFloat32.item
            .sum / numBatchesPerValidation
          println(s"Validation cost $step: ${avgValLoss}")
          logger.save(state, step)
          println(s"Checkpoint saved")
          println("-" * 30)
      .drop(1_000_000_000)
      .next()
