package deepwit.examples.autoencoder

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.optimizer.Adam
import dimwit.tensortree.{TensorTree, TensorTreeIO}

import deepwit.examples.dataset.MNISTLoader
import deepwit.loss.BinaryCrossEntropy

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** Steady-state cost of one autoencoder training step, JIT-compiled and eager.
  *
  * The timed region is exactly one `gradientStep` — value-and-grad of the batch cost,
  * then the Adam update — on a batch that is already resident on the device. Loading
  * and batching sit outside it. `examples/src/main/python/autoencoder_bench.py` runs the
  * same protocol over a hand-written JAX program of the same shape, so the two numbers
  * are comparable.
  *
  * Run as, from the repository root:
  * {{{
  * sbt "examples/runMain deepwit.examples.autoencoder.benchAutoEncoder steps=100 reps=10"
  * }}}
  */
object AutoEncoderBench:

  trait Batch derives Label

  // -- Configuration, mirroring AutoEncoderTrain --

  val batchSize = 256
  val latentDim = 24
  val learningRate = 3e-4f

  val eHidden1Extent = Axis[EHidden1] -> 512
  val eHidden2Extent = Axis[EHidden2] -> 256
  val latentExtent = Axis[Latent] -> latentDim
  val dHidden1Extent = Axis[DHidden1] -> 256
  val dHidden2Extent = Axis[DHidden2] -> 512

  val optimizer = Adam(learningRate = learningRate)

  // -- The step under measurement, character for character the one in AutoEncoderTrain --

  def costFnFor[S: Label](samples: Tensor3[S, Height, Width, Float32])(params: Autoencoder.Params): Tensor0[Float32] =
    val model = Autoencoder(params)
    samples
      .vmap(Axis[S]): sample =>
        val original = sample.flatten
        zipvmap(Axis[Pixel])(original, model.logits(original)): (origPixel, reconPixel) =>
          BinaryCrossEntropy.fromLogits(origPixel, reconPixel)
        .sum
      .mean

  def gradientStep(batch: Tensor3[Batch, Height, Width, Float32], state: TrainState): TrainState =
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, cost)

  def initialState(key: Key): TrainState =
    val params = Autoencoder.Params.init(eHidden1Extent, eHidden2Extent, latentExtent, dHidden1Extent, dHidden2Extent, key)
    TrainState(params, optimizer.init(params), Tensor0(-1f))

  // -- Measurement --

  /** JAX dispatch is asynchronous, so a timed region only means something once every
    * array it produced has actually landed.
    */
  def blockUntilReady[T: TensorTree](tree: T): T =
    Jax.jax.block_until_ready(TensorTree[T].toPyTree(tree))
    tree

  case class Measurement(label: String, firstCallMs: Double, stepMsPerRep: Seq[Double], stepsPerRep: Int):
    private val sorted = stepMsPerRep.sorted
    val reps: Int = stepMsPerRep.size
    val mean: Double = stepMsPerRep.sum / reps
    val std: Double =
      if reps < 2 then 0.0
      else math.sqrt(stepMsPerRep.map(t => (t - mean) * (t - mean)).sum / (reps - 1))
    val median: Double = if reps % 2 == 1 then sorted(reps / 2) else (sorted(reps / 2 - 1) + sorted(reps / 2)) / 2
    val min: Double = sorted.head
    val max: Double = sorted.last
    val stepsPerSecond: Double = 1000.0 / mean

    def render: String =
      f"$label%-28s ${mean}%9.3f ± ${std}%-8.3f ${median}%9.3f ${min}%9.3f ${max}%9.3f ${stepsPerSecond}%9.1f ${firstCallMs}%10.1f"

    def toJson: String =
      s"""    {
         |      "label": "$label",
         |      "first_call_ms": $firstCallMs,
         |      "steps_per_rep": $stepsPerRep,
         |      "reps": $reps,
         |      "mean_step_ms": $mean,
         |      "std_step_ms": $std,
         |      "median_step_ms": $median,
         |      "min_step_ms": $min,
         |      "max_step_ms": $max,
         |      "steps_per_second": $stepsPerSecond,
         |      "step_ms_per_rep": [${stepMsPerRep.mkString(", ")}]
         |    }""".stripMargin

  /** One measurement: a discarded first call, `warmup` discarded steps, then `reps`
    * timed runs of `stepsPerRep` steps each. Each run is blocked on at its end, inside
    * the timed region, so no work escapes the clock.
    *
    * `initial` is by-name because the donating JIT step deletes the buffers it is handed,
    * so every measurement has to be given a state of its own.
    */
  def measure[B, S: TensorTree](
      label: String,
      step: (B, S) => S,
      batch: B,
      initial: => S,
      lastCost: S => Tensor0[Float32],
      warmup: Int,
      reps: Int,
      stepsPerRep: Int
  ): Measurement =
    var state = blockUntilReady(initial)

    val firstCallStart = System.nanoTime()
    state = blockUntilReady(step(batch, state))
    val firstCallMs = (System.nanoTime() - firstCallStart) / 1e6

    for _ <- 1 to warmup do state = step(batch, state)
    state = blockUntilReady(state)

    val stepMsPerRep =
      for _ <- 1 to reps yield
        val start = System.nanoTime()
        for _ <- 1 to stepsPerRep do state = step(batch, state)
        state = blockUntilReady(state)
        (System.nanoTime() - start) / 1e6 / stepsPerRep

    // Keep the trajectory honest: a diverged run would have been measuring NaNs.
    val finalCost = lastCost(state).item
    require(!finalCost.isNaN, s"$label diverged to NaN")

    Measurement(label, firstCallMs, stepMsPerRep, stepsPerRep)

  /** The loss over the first `steps` steps, for checking the Python program computes
    * the same thing rather than merely something of the same shape.
    */
  def lossTrace[B, S](step: (B, S) => S, batch: B, initial: => S, lastCost: S => Tensor0[Float32], steps: Int): Seq[Float] =
    var state = initial
    for _ <- 1 to steps yield
      state = step(batch, state)
      lastCost(state).item

  /** Cost of marshalling one state out to Python and back again.
    *
    * This is what a JIT call pays on every invocation on top of running the executable:
    * `toPyTree` of the arguments on the way in, `fromPyTree` of the result on the way out.
    */
  def roundTripMs[S: TensorTree](state: S, iterations: Int): Double =
    val tree = TensorTree[S]
    for _ <- 1 to 20 do tree.fromPyTree(tree.toPyTree(state))
    val start = System.nanoTime()
    for _ <- 1 to iterations do tree.fromPyTree(tree.toPyTree(state))
    (System.nanoTime() - start) / 1e6 / iterations

  /** How many tensors a state is made of, which is what the marshalling cost scales with. */
  def leafCount[S: TensorTree](state: S): Int =
    TensorTree[S].mapLeaves(state, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => 1).sum

@main
def benchAutoEncoder(args: String*): Unit =
  import AutoEncoderBench.*

  val opts = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
  val stepsPerRep = opts.getOrElse("steps", "100").toInt
  val reps = opts.getOrElse("reps", "10").toInt
  val warmup = opts.getOrElse("warmup", "20").toInt
  val eagerStepsPerRep = opts.getOrElse("eagerSteps", "20").toInt
  val traceSteps = opts.getOrElse("trace", "20").toInt
  val outDir = Path.of(opts.getOrElse("out", "out/bench"))
  val modes = opts.getOrElse("mode", "both")

  Files.createDirectories(outDir)

  val device = Jax.devices.head
  println(s"Device: $device")
  println(s"JAX:    ${Jax.jax.__version__}")

  // -- The batch under measurement: the first batch of the training stream, on device --

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val batch = trainDataset.toBatchStream(Axis[Batch] -> batchSize).map(_.images).next()
  blockUntilReady(batch)

  val initKey = Key(42)

  // Same starting point for the Python program, so the loss traces are comparable.
  TensorTreeIO.save(initialState(initKey).params, outDir.resolve("init_params.pkl"))

  val batchChecksum = batch.sum.item
  println(f"Batch checksum (sum of pixels): $batchChecksum%.6f")

  // Measured before the loss trace, so that the first call of each configuration is a
  // genuinely cold compile rather than one served from a cache the trace has already filled.
  val measurements =
    val jitted =
      if modes == "eager" then Seq.empty
      else
        val jitStep = jitDonatingUnsafe(gradientStep)
        Seq(measure("deepwit-jit", jitStep, batch, initialState(initKey), _.lastCost, warmup, reps, stepsPerRep))
    val eager =
      if modes == "jit" then Seq.empty
      else Seq(measure("deepwit-eager", gradientStep, batch, initialState(initKey), _.lastCost, warmup, reps, eagerStepsPerRep))
    jitted ++ eager

  println()
  println(f"${"configuration"}%-28s ${"mean ms"}%9s   ${"std"}%-8s ${"median"}%9s ${"min"}%9s ${"max"}%9s ${"steps/s"}%9s ${"1st call ms"}%10s")
  measurements.foreach(m => println(m.render))

  val json =
    s"""{
       |  "implementation": "deepwit",
       |  "device": "$device",
       |  "jax_version": "${Jax.jax.__version__}",
       |  "batch_size": $batchSize,
       |  "latent_dim": $latentDim,
       |  "learning_rate": $learningRate,
       |  "warmup_steps": $warmup,
       |  "batch_checksum": ${batchChecksum.toDouble},
       |  "measurements": [
       |${measurements.map(_.toJson).mkString(",\n")}
       |  ]
       |}
       |""".stripMargin
  val trace = lossTrace(jitDonatingUnsafe(gradientStep), batch, initialState(initKey), _.lastCost, traceSteps)
  Files.writeString(
    outDir.resolve("scala_loss_trace.csv"),
    ("step,loss" +: trace.zipWithIndex.map { case (l, i) => s"${i + 1},${l.toDouble}" }).mkString("\n") + "\n",
    StandardCharsets.UTF_8
  )
  println(s"Loss trace: ${trace.head} .. ${trace.last}")

  Files.writeString(outDir.resolve("deepwit_bench.json"), json, StandardCharsets.UTF_8)
  println(s"\nWrote ${outDir.toAbsolutePath}/deepwit_bench.json")

/** The six-layer autoencoder swept over batch size.
  *
  * The depth sweep moves work and leaf count together. Batch size moves only the work: the
  * parameter tree is the same 39 tensors at every point here, so DeepWit's per-call bridge cost
  * is a constant that a larger batch has more computation to hide behind. This is the axis on
  * which the two implementations should converge.
  *
  * Run, from the repository root:
  * {{{
  * sbt "examples/runMain deepwit.examples.autoencoder.benchAutoEncoderBatch batches=64,256,1024,4096,16384"
  * }}}
  */
@main
def benchAutoEncoderBatch(args: String*): Unit =
  import AutoEncoderBench.*

  val opts = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
  val batchSizes = opts.getOrElse("batches", "64,256,1024,4096,16384").split(",").map(_.trim.toInt).toSeq
  val stepsPerRep = opts.getOrElse("steps", "100").toInt
  val reps = opts.getOrElse("reps", "10").toInt
  val warmup = opts.getOrElse("warmup", "20").toInt
  val eagerStepsPerRep = opts.getOrElse("eagerSteps", "5").toInt
  val eagerReps = opts.getOrElse("eagerReps", "8").toInt
  val outDir = Path.of(opts.getOrElse("out", "out/bench"))
  val modes = opts.getOrElse("mode", "both")

  Files.createDirectories(outDir)

  val device = Jax.devices.head
  println(s"Device: $device")
  println(s"JAX:    ${Jax.jax.__version__}")

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val initKey = Key(42)

  // Measured once: the parameter tree does not depend on the batch, which is the point.
  val probe = blockUntilReady(initialState(initKey))
  val leaves = leafCount(probe)
  val bridgeMs = roundTripMs(probe, 100)
  println(f"Train state: $leaves%d tensors, bridge round trip ${bridgeMs}%.3f ms")

  // Same starting point for the Python program, so the two are comparable numerically.
  TensorTreeIO.save(initialState(initKey).params, outDir.resolve("init_params.pkl"))

  val results =
    for batchSize <- batchSizes yield
      val batch = blockUntilReady(trainDataset.toBatchStream(Axis[Batch] -> batchSize).map(_.images).next())
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
       |  "model": "autoencoder-6layer",
       |  "device": "$device",
       |  "jax_version": "${Jax.jax.__version__}",
       |  "latent_dim": ${AutoEncoderBench.latentDim},
       |  "learning_rate": ${AutoEncoderBench.learningRate},
       |  "warmup_steps": $warmup,
       |  "leaves": $leaves,
       |  "round_trip_ms": $bridgeMs,
       |  "configs": [
       |${configsJson.mkString(",\n")}
       |  ]
       |}
       |""".stripMargin
  Files.writeString(outDir.resolve("deepwit_batch_bench.json"), json, StandardCharsets.UTF_8)
  println(s"\nWrote ${outDir.toAbsolutePath}/deepwit_batch_bench.json")
