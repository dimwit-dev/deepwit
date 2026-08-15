package deepwit.training

/** Renders a one-line report about the training state at a given step. */
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
