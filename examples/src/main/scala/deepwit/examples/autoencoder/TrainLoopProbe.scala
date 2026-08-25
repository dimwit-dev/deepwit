package deepwit.examples.autoencoder

import dimwit.*
import dimwit.jax.Jax

import deepwit.examples.dataset.MNISTLoader
import AutoEncoderBench.{Batch, blockUntilReady, gradientStep, initialState}

/** What one iteration of `AutoEncoderTrain`'s trajectory actually costs.
  *
  * The benchmarks feed a pre-materialised batch, so they measure the gradient step alone. The
  * training loop also pulls a batch from `toBatchStream` on every iteration, which gathers rows
  * on device from a Scala `Seq[Int]` of indices. This times the two separately and together, so
  * the data pipeline can be told apart from the step.
  *
  * Run, from the repository root:
  * {{{
  * sbt "examples/runMain deepwit.examples.autoencoder.trainLoopProbe batch=512"
  * }}}
  */
@main
def trainLoopProbe(args: String*): Unit =
  val opts = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
  val batchSize = opts.getOrElse("batch", "512").toInt
  val steps = opts.getOrElse("steps", "100").toInt
  val warmup = opts.getOrElse("warmup", "20").toInt

  println(s"Device: ${Jax.devices.head}")
  println(s"Batch $batchSize, $steps steps per measurement\n")

  val dataset = MNISTLoader.createTrainingDataset().get
  def freshStream = dataset.toBatchStream(Axis[Batch] -> batchSize).map(_.images)

  def timed(label: String)(body: Int => Unit): Double =
    body(warmup)
    val start = System.nanoTime()
    body(steps)
    val ms = (System.nanoTime() - start) / 1e6 / steps
    println(f"$label%-52s ${ms}%8.3f ms/iteration")
    ms

  // Compiled once, outside every timed region: `timed` runs its body twice, so a jit created
  // inside would be recompiled and the compile would land in the measurement.
  val jitStep = jitDonatingUnsafe(gradientStep)
  val fixedBatch = blockUntilReady(freshStream.next())
  blockUntilReady(jitStep(fixedBatch, initialState(Key(42))))

  // 1. The data pipeline on its own: one `next()` per iteration.
  val dataMs = timed("1. batch stream only (toBatchStream.next)"): n =>
    val stream = freshStream
    for _ <- 1 to n do blockUntilReady(stream.next())

  // 2. The gradient step on its own, on one fixed batch — what the benchmarks measure.
  val stepMs = timed("2. jitted gradient step only (fixed batch)"): n =>
    var state = blockUntilReady(initialState(Key(42)))
    for _ <- 1 to n do state = jitStep(fixedBatch, state)
    blockUntilReady(state)

  // 3. Both, exactly as `trainTrajectory` runs them.
  val loopMs = timed("3. full trajectory iteration (stream + step)"): n =>
    val stream = freshStream
    var state = blockUntilReady(initialState(Key(42)))
    for _ <- 1 to n do state = jitStep(stream.next(), state)
    blockUntilReady(state)

  println()
  println(f"data pipeline share of one iteration: ${100 * dataMs / loopMs}%.1f%%")
  println(f"step + data, measured separately:     ${dataMs + stepMs}%8.3f ms")
  println(f"measured together:                    ${loopMs}%8.3f ms  (overlap saves ${dataMs + stepMs - loopMs}%.3f ms)")
