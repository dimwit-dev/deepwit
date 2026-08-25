package deepwit.examples.mnistClassification

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.optimizer.GradientDescent
import dimwit.tensortree.TensorTreeIO

import deepwit.loss.CategoricalCrossEntropy
import deepwit.examples.dataset.{MNISTLoader, MNISTBatchSample}
import deepwit.examples.autoencoder.AutoEncoderBench.{blockUntilReady, leafCount, lossTrace, measure, roundTripMs}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** The autoencoder batch sweep, repeated on the MNIST CNN as an independent check.
  *
  * A different architecture (convolutions rather than dense layers), a different optimizer
  * (plain gradient descent, whose state is `Unit`), and a much smaller parameter tree — seven
  * tensors against the autoencoder's thirty-nine. If DeepWit's per-call cost really is
  * proportional to leaf count, this model should carry roughly a seventh of the autoencoder's
  * fixed host cost and therefore reach parity with JAX at a far smaller batch.
  *
  * Run, from the repository root:
  * {{{
  * sbt "examples/runMain deepwit.examples.mnistClassification.benchMNistCNN batches=128,512,2048,8192"
  * }}}
  */
object MNistCNNBench:

  trait Batch derives Label

  val learningRate = 0.01f
  val numHidden1 = 16
  val numHidden2 = 32

  val optimizer = GradientDescent(learningRate = learningRate)

  def costFnFor[S: Label](images: Tensor3[S, Height, Width, Float32], labels: Tensor1[S, Int32])(params: MNistCNN.Params): Tensor0[Float32] =
    val model = MNistCNN(params)
    zipvmap(Axis[S])(images, labels): (image, label) =>
      val logits = model.logits(image)
      CategoricalCrossEntropy.fromLogits(label, logits)
    .mean

  def gradientStep(batch: MNISTBatchSample[Batch], state: TrainState): TrainState =
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch.images, batch.labels))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, cost)

  def initialState(key: Key): TrainState =
    val params = MNistCNN.Params.init(numHidden1, numHidden2, key)
    TrainState(params, optimizer.init(params), Tensor0(-1f))

@main
def benchMNistCNN(args: String*): Unit =
  import MNistCNNBench.*

  val opts = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
  val batchSizes = opts.getOrElse("batches", "128,512,2048,8192").split(",").map(_.trim.toInt).toSeq
  val stepsPerRep = opts.getOrElse("steps", "100").toInt
  val reps = opts.getOrElse("reps", "10").toInt
  val warmup = opts.getOrElse("warmup", "20").toInt
  val eagerStepsPerRep = opts.getOrElse("eagerSteps", "5").toInt
  val eagerReps = opts.getOrElse("eagerReps", "8").toInt
  val traceSteps = opts.getOrElse("trace", "20").toInt
  val outDir = Path.of(opts.getOrElse("out", "out/bench/cnn"))
  val modes = opts.getOrElse("mode", "both")

  Files.createDirectories(outDir)

  val device = Jax.devices.head
  println(s"Device: $device")
  println(s"JAX:    ${Jax.jax.__version__}")

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val initKey = Key(42)

  val probe = blockUntilReady(initialState(initKey))
  val leaves = leafCount(probe)
  val bridgeMs = roundTripMs(probe, 100)
  println(f"Train state: $leaves%d tensors, bridge round trip ${bridgeMs}%.3f ms")

  // Same starting point for the Python program.
  TensorTreeIO.save(initialState(initKey).params, outDir.resolve("cnn_init_params.pkl"))

  val results =
    for batchSize <- batchSizes yield
      val batch = blockUntilReady(trainDataset.toBatchStream(Axis[Batch] -> batchSize).next())
      println(s"\n== batch $batchSize ==")
      val measurements =
        val jitted =
          if modes == "eager" then Seq.empty
          else
            val jitStep = jitDonatingUnsafe(gradientStep)
            Seq(measure(s"deepwit-jit-b$batchSize", jitStep, batch, initialState(initKey), _.lastCost, warmup, reps, stepsPerRep))
        val eager =
          if modes == "jit" then Seq.empty
          else Seq(measure(s"deepwit-eager-b$batchSize", gradientStep, batch, initialState(initKey), _.lastCost, warmup, eagerReps, eagerStepsPerRep))
        jitted ++ eager
      measurements.foreach(m => println(m.render))
      Jax.gc()
      (batchSize, measurements)

  // Loss trace at the example's own batch size, for the equivalence check.
  val traceBatch = blockUntilReady(trainDataset.toBatchStream(Axis[Batch] -> 128).next())
  val batchChecksum = traceBatch.images.sum.item
  val trace = lossTrace(jitDonatingUnsafe(gradientStep), traceBatch, initialState(initKey), _.lastCost, traceSteps)
  Files.writeString(
    outDir.resolve("cnn_scala_loss_trace.csv"),
    ("step,loss" +: trace.zipWithIndex.map { case (l, i) => s"${i + 1},${l.toDouble}" }).mkString("\n") + "\n",
    StandardCharsets.UTF_8
  )
  println(f"\nBatch checksum (sum of pixels, batch 128): ${batchChecksum.toDouble}%.6f")
  println(s"Loss trace: ${trace.head} .. ${trace.last}")

  println()
  println(f"${"configuration"}%-28s ${"mean ms"}%9s   ${"std"}%-8s ${"median"}%9s ${"min"}%9s ${"max"}%9s ${"steps/s"}%9s ${"1st call ms"}%10s")
  results.foreach(_._2.foreach(m => println(m.render)))

  val configsJson = results.map: (batchSize, measurements) =>
    s"""    {
       |      "batch_size": $batchSize,
       |      "measurements": [
       |${measurements.map(_.toJson).mkString(",\n")}
       |      ]
       |    }""".stripMargin

  val json =
    s"""{
       |  "implementation": "deepwit",
       |  "model": "mnist-cnn",
       |  "device": "$device",
       |  "jax_version": "${Jax.jax.__version__}",
       |  "learning_rate": $learningRate,
       |  "warmup_steps": $warmup,
       |  "leaves": $leaves,
       |  "round_trip_ms": $bridgeMs,
       |  "batch_checksum": ${batchChecksum.toDouble},
       |  "configs": [
       |${configsJson.mkString(",\n")}
       |  ]
       |}
       |""".stripMargin
  Files.writeString(outDir.resolve("deepwit_batch_bench.json"), json, StandardCharsets.UTF_8)
  println(s"\nWrote ${outDir.toAbsolutePath}/deepwit_batch_bench.json")
