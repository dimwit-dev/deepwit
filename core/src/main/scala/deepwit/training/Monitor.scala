package deepwit.training

import dimwit.*

import deepwit.optimizer.LearningRateSchedule

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

  /** Reports the learning rate the schedule prescribes for the step. */
  case class LearningRateMonitor[S](schedule: LearningRateSchedule) extends Monitor[S]:
    def report(step: Int, state: S): String =
      f"LR: ${schedule(Tensor0(step)).item}%.2e"

  /** Reports how many units of work per second the run sustains, given how many of them a step processes.
    *
    * The unit is whatever a run measures its progress in — samples, tokens, frames, ... — and is named
    * in the report as `<unitName>/sec`. Not a case class: it carries the timing of the previous report,
    * so that it can be specialised to a unit by extending it.
    */
  class ThroughputMonitor[S](unitsPerStep: Int, unitName: String) extends Monitor[S]:
    private var lastTime = System.nanoTime()
    private var lastStep = 0

    def report(step: Int, state: S): String =
      require(step >= lastStep, "Step must be non-decreasing.")
      val elapsedSteps = step - lastStep
      lastStep = step
      val currentTime = System.nanoTime()
      val elapsedS = (currentTime - lastTime) / 1e9d
      val unitsPerSec = (unitsPerStep.toDouble * elapsedSteps) / elapsedS
      lastTime = currentTime
      f"$unitsPerSec%.2f $unitName/sec"

  /** Reports sample throughput, the [[ThroughputMonitor]] every training loop wants. */
  class PerformanceMonitor[S](batchSize: Int) extends ThroughputMonitor[S](batchSize, "samples")
