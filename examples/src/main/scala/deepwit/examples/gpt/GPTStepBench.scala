package deepwit.examples.gpt

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.optimizer.{Adam, AdamState}
import dimwit.tensortree.TensorTree

import deepwit.loss.CategoricalCrossEntropy
import deepwit.examples.autoencoder.AutoEncoderBench.{blockUntilReady, leafCount, measure, roundTripMs}

import FineWebDataset.BatchSample

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** What one GPT-2 training step costs, and how much of it is the ScalaPy bridge.
  *
  * The autoencoder benchmarks establish that DeepWit's per-call overhead is proportional to
  * the number of tensors in the train state, not to the work the step does. That predicts the
  * overhead is invisible on a model whose step is long — which is the regime GPT-2 training
  * runs in. This measures both quantities on the real GPT-2 parameters so the prediction can
  * be checked rather than assumed.
  *
  * Tokens are synthetic, so no FineWeb shard is needed: the step's cost depends on the shapes,
  * not on which tokens are in the batch.
  *
  * Run, from the repository root:
  * {{{
  * sbt "examples/runMain deepwit.examples.gpt.benchGPTStep batch=4 context=1024"
  * }}}
  */
object GPTStepBench:

  val optimizer = Adam(learningRate = 6e-4f)

  case class GPTTrainState(
      params: GPT.Params[Float32],
      optimizerState: AdamState[GPT.Params[Float32]],
      lastCost: Tensor0[Float32]
  )

  def loss(targets: Tensor1[Context, Int32], logits: Tensor2[Context, Vocab, Float32]): Tensor0[Float32] =
    zipvmap(Axis[Context])(targets, logits)(CategoricalCrossEntropy.fromLogits).mean

  def costFnFor(batchSample: BatchSample)(params: GPT.Params[Float32]): Tensor0[Float32] =
    val model = GPT(params)
    zipvmap(Axis[Sample])(batchSample.targets, batchSample.inputs):
      case (targets, inputs) => loss(targets, model.logits(inputs))
    .mean

  def gradientStep(batchSample: BatchSample, state: GPTTrainState): GPTTrainState =
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batchSample))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    GPTTrainState(newParams, newOptimizerState, cost)

  /** A batch of uniformly random token ids. Only the shapes reach the compiled step. */
  def syntheticBatch(batchSize: Int, contextLength: Int, vocabSize: Int, seed: Int): BatchSample =
    val random = scala.util.Random(seed)
    val shape = Shape(Axis[Sample] -> batchSize, Axis[Context] -> contextLength)
    def tokens = Tensor(shape, VType[Int32]).fromArray(Array.fill(batchSize * contextLength)(random.nextInt(vocabSize)))
    BatchSample(targets = tokens, inputs = tokens)

@main
def benchGPTStep(args: String*): Unit =
  import GPTStepBench.*

  val opts = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
  val batchSizes = opts.getOrElse("batches", opts.getOrElse("batch", "4")).split(",").map(_.trim.toInt).toSeq
  val contextLength = opts.getOrElse("context", "1024").toInt
  val numLayers = opts.getOrElse("layers", "12").toInt
  val numHeads = opts.getOrElse("heads", "12").toInt
  val vocabSize = opts.getOrElse("vocab", "50304").toInt
  val stepsPerRep = opts.getOrElse("steps", "5").toInt
  val reps = opts.getOrElse("reps", "5").toInt
  val warmup = opts.getOrElse("warmup", "3").toInt
  val outDir = Path.of(opts.getOrElse("out", "out/bench"))

  Files.createDirectories(outDir)

  println(s"Device: ${Jax.devices.head}")
  println(s"JAX:    ${Jax.jax.__version__}")
  println(s"GPT-2: $numLayers layers, $numHeads heads, vocab $vocabSize, context $contextLength, batches ${batchSizes.mkString(",")}")

  val vocabExtent = Axis[Vocab] -> vocabSize
  val contextExtent = Axis[Context] -> contextLength
  val embeddingExtent = Axis[Embedding] -> (64 * numHeads)
  val embeddingMixedExtent = Axis[EmbeddingMixed] -> (4 * 64 * numHeads)

  def initialState(): GPTTrainState =
    val params = GPT.Params.gpt2Init[Float32](numTransformerLayers = numLayers)(
      vocabExtent,
      contextExtent,
      numHeads,
      embeddingExtent,
      embeddingMixedExtent,
      Key(42),
      VType[Float32]
    )
    GPTTrainState(params, optimizer.init(params), Tensor0(-1f))

  // Measured once: the bridge cost is set by how many tensors the state holds, which does
  // not change with the batch. That is the whole point of the sweep.
  val probe = blockUntilReady(initialState())
  val leaves = leafCount(probe)
  val bridgeMs = roundTripMs(probe, 20)
  println(f"Train state: $leaves%d tensors, bridge round trip ${bridgeMs}%.3f ms")

  val results =
    for batchSize <- batchSizes yield
      scala.util.Try:
        val batch = blockUntilReady(syntheticBatch(batchSize, contextLength, vocabSize, seed = 7))
        val jitStep = jitDonatingUnsafe(gradientStep)
        val m = measure(s"deepwit-jit-gpt2-b$batchSize", jitStep, batch, initialState(), _.lastCost, warmup, reps, stepsPerRep)
        println(m.render)
        Jax.gc()
        (batchSize, m)
      .recover:
        case e: Throwable =>
          println(s"batch $batchSize: did not fit (${e.getClass.getSimpleName})")
          Jax.gc()
          null
      .toOption
      .flatMap(Option(_))

  val ok = results.flatten

  println()
  println(f"${"batch"}%6s ${"tokens"}%8s ${"step ms"}%10s ${"bridge ms"}%10s ${"bridge %"}%9s ${"predicted DeepWit/JAX"}%22s")
  ok.foreach: (batchSize, m) =>
    val tokens = batchSize * contextLength
    val predicted = m.mean / (m.mean - bridgeMs)
    println(f"$batchSize%6d $tokens%8d ${m.mean}%10.2f ${bridgeMs}%10.2f ${100.0 * bridgeMs / m.mean}%8.1f%% ${predicted}%21.2fx")

  val json =
    s"""{
       |  "implementation": "deepwit",
       |  "model": "gpt2",
       |  "device": "${Jax.devices.head}",
       |  "jax_version": "${Jax.jax.__version__}",
       |  "layers": $numLayers,
       |  "heads": $numHeads,
       |  "vocab": $vocabSize,
       |  "context": $contextLength,
       |  "batch_sizes": [${ok.map(_._1).mkString(", ")}],
       |  "leaves": $leaves,
       |  "round_trip_ms": $bridgeMs,
       |  "measurements": [
       |${ok.map(_._2.toJson).mkString(",\n")}
       |  ]
       |}
       |""".stripMargin
  Files.writeString(outDir.resolve("deepwit_gpt_bench.json"), json, StandardCharsets.UTF_8)
  println(s"\nWrote ${outDir.toAbsolutePath}/deepwit_gpt_bench.json")
