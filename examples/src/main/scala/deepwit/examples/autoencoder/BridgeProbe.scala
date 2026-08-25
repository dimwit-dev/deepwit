package deepwit.examples.autoencoder

import dimwit.*
import dimwit.jax.Jax
import dimwit.tensortree.TensorTree

import deepwit.examples.mnistClassification.MNistCNNBench

import me.shadaj.scalapy.py

/** Where does a DeepWit step's host time actually go?
  *
  * One compiled JAX function, called three ways with the same arguments, so the per-call
  * cost can be attributed instead of assumed:
  *
  *   1. from Python with Python-resident arguments — the JAX dispatch baseline;
  *   2. from Scala with the *same* Python-resident arguments — adds only ScalaPy's call
  *      mechanism, and no tree work at all;
  *   3. from Scala through `TensorTree.toPyTree` / `fromPyTree` — what `Jit.toPyJit` does.
  *
  * (2) - (1) is what crossing the JVM/CPython boundary costs once per call. (3) - (2) is what
  * rebuilding the tree costs. Run over both benchmarked examples, whose train states differ by
  * more than five times in tensor count, so a per-leaf term can be told apart from a per-call one.
  *
  * Run, from the repository root:
  * {{{
  * sbt "examples/runMain deepwit.examples.autoencoder.bridgeProbe steps=200"
  * }}}
  */
@main
def bridgeProbe(args: String*): Unit =
  val opts = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
  val steps = opts.getOrElse("steps", "200").toInt
  val pythonDir = opts.getOrElse("pythonDir", "src/main/python")

  println(s"Device: ${Jax.devices.head}")
  println(s"JAX:    ${Jax.jax.__version__}")

  py.module("sys").path.applyDynamic("insert")(0, pythonDir)
  val probe = py.module("bridge_probe")

  /** The three loops, over one tree. */
  def report[S: TensorTree](label: String, scalaState: S): Unit =
    val tree = TensorTree[S]
    val pyState0 = tree.toPyTree(scalaState).as[py.Dynamic]
    val leaves = probe.leaf_count(pyState0).as[Int]
    println(s"\n== $label: $leaves tensors, $steps steps per measurement ==")

    def timed(name: String)(body: Int => Unit): Double =
      body(20)
      val start = System.nanoTime()
      body(steps)
      val ms = (System.nanoTime() - start) / 1e6 / steps
      println(f"$name%-54s ${ms}%8.3f ms/step")
      ms

    val pythonFirstMs = probe.time_python(pyState0, steps).as[Double]
    println(f"${"1. Python loop, Python-resident args (JAX baseline)"}%-54s ${pythonFirstMs}%8.3f ms/step")

    val scalaCallMs = timed("2. Scala loop, Python-resident args (+ ScalaPy call)"): n =>
      var s = pyState0
      for _ <- 1 to n do s = probe.step(s).as[py.Dynamic]
      probe.block(s)

    val scalaTreeMs = timed("3. Scala loop, toPyTree/fromPyTree (full DeepWit path)"): n =>
      var s = scalaState
      for _ <- 1 to n do s = tree.fromPyTree(probe.step(tree.toPyTree(s)).as[py.Dynamic])
      probe.block(tree.toPyTree(s))

    // Repeated last, to check the baseline is not an artefact of running first.
    val pythonLastMs = probe.time_python(pyState0, steps).as[Double]
    println(f"${"1b. Python loop again, after the Scala loops"}%-54s ${pythonLastMs}%8.3f ms/step")
    val pythonMs = math.min(pythonFirstMs, pythonLastMs)

    val callOverhead = scalaCallMs - pythonMs
    val treeOverhead = scalaTreeMs - scalaCallMs
    val total = scalaTreeMs - pythonMs
    println(f"   ScalaPy call overhead   (2-1): ${callOverhead}%8.3f ms  (${1000 * callOverhead / leaves}%6.1f us/leaf)")
    println(f"   dimwit tree marshalling (3-2): ${treeOverhead}%8.3f ms  (${1000 * treeOverhead / leaves}%6.1f us/leaf)")
    println(f"   total DeepWit overhead  (3-1): ${total}%8.3f ms  (${1000 * total / leaves}%6.1f us/leaf)")
    println(f"   share that is tree marshalling: ${100 * treeOverhead / total}%.1f%%")

  report("six-layer autoencoder", AutoEncoderBench.initialState(Key(42)))
  report("MNIST CNN", MNistCNNBench.initialState(Key(42)))
