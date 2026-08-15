package object deepwit:

  trait Monitor[S]:
    def report(step: Int, state: S): String

  object Monitor:

    def default[S](batchSize: Int, lossLens: S => Float): Monitor[S] =
      ConcatMonitor(List(
        StepMonitor(),
        LossMonitor(lossLens),
        PerformanceMonitor(batchSize)
      ))

    case class ConcatMonitor[S](monitors: List[Monitor[S]], sep: String = " | ") extends Monitor[S]:
      def report(step: Int, state: S): String =
        monitors.map(m => m.report(step, state)).mkString(sep)

    case class StepMonitor[S]() extends Monitor[S]:
      def report(step: Int, state: S): String =
        f"Step $step"

    case class LossMonitor[S](lossLens: S => Float) extends Monitor[S]:
      def report(step: Int, state: S): String =
        f"Loss: ${lossLens(state)}%.4f"

    case class PerformanceMonitor[S](batchSize: Int) extends Monitor[S]:
      private var lastTime = System.nanoTime()
      private var lastStep = 0

      def report(step: Int, state: S): String =
        require(step >= lastStep, "Step must be non-decreasing.")
        val elapsedSteps = step - lastStep
        lastStep = step
        val currentTime = System.nanoTime()
        val elapsedS = (currentTime - lastTime) / 1e9d
        val sPerSec = (batchSize * elapsedSteps) / elapsedS
        lastTime = currentTime
        f"$sPerSec%.2f samples/sec"

  extension [T](it: Iterator[T])

    def tapEvery(n: Int)(f: (T, Int) => Unit): Iterator[T] =
      it
        .zipWithIndex
        .tapEach: (t, id) =>
          if id > 0 && id % n == 0 then f(t, id)
        .map(_._1)

  extension [T](it: LazyList[T])

    def tapEvery(n: Int)(f: (T, Int) => Unit): LazyList[T] =
      it
        .zipWithIndex
        .tapEach: (t, id) =>
          if id > 0 && id % n == 0 then f(t, id)
        .map(_._1)

  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.TreeOf
  import dimwit.TreeOf.{map, mapLeaves}

  extension [H: TensorTree, V: IsFloating](grads: Grad[H])(using TreeOf[H, V])
    def clipGlobalNorm(maxNorm: Tensor0[V], epsilon: Double = 1e-6): Grad[H] =
      val gradNorm = grads.value.mapLeaves([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x.pow(2).sum).reduce(_ + _).sqrt
      val scale = minimum(1f, maxNorm / (gradNorm + epsilon))
      Grad(grads.value.map([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x *! scale))

  private[deepwit] def defaultEpsilon(dtype: DType): Float = dimwit.jax.Jax.jnp.finfo(dtype.jaxType).eps.as[Float]
  private[deepwit] def unwrapEpsilon(eps: Float | (DType => Float), dtype: DType): Float = eps match
    case ε: Float            => ε
    case f: (DType => Float) => f(dtype)
