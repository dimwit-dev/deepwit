package deepwit.optimizer.schedule

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.GradientOptimizer

case class LearningRateSchedulerState[P, State[_]](
    step: Tensor0[Int32],
    optState: State[P]
)
type LearningRateSchedulerStateFor[State[_]] = [P] =>> LearningRateSchedulerState[P, State]

class LearningRateScheduler[State[_]](val optF: Tensor0[Float32] => GradientOptimizer[State], schedule: Tensor0[Int32] => Tensor0[Float32]) extends GradientOptimizer[LearningRateSchedulerStateFor[State]]:

  def init[P: TensorTree, V](params: P)(using TreeOf[P, V])(using IsFloating[V]): LearningRateSchedulerState[P, State] =
    val step = Tensor0(1)
    val opt = optF(schedule(step))
    LearningRateSchedulerState(step, opt.init(params))

  def update[P: TensorTree, V](gradients: Grad[P], params: P, state: LearningRateSchedulerState[P, State])(using TreeOf[P, V])(using IsFloating[V]): (P, LearningRateSchedulerState[P, State]) =
    val step = state.step
    val optState = state.optState
    val opt = optF(schedule(step))
    val (newParams, newOptState) = opt.update(gradients, params, optState)
    (newParams, LearningRateSchedulerState(step + 1, newOptState))

trait LearningRateSchedule(val steps: Tensor0[Int32]) extends (Tensor0[Int32] => Tensor0[Float32])

object LearningRateSchedule:

  private class FollowBySchedule(first: LearningRateSchedule, second: LearningRateSchedule)
      extends LearningRateSchedule(steps = first.steps + second.steps):
    override def apply(step: Tensor0[Int32]): Tensor0[Float32] =
      val firstValue = first(step)
      val secondValue = second.delay(first.steps)(step)
      where(step <= first.steps, firstValue, secondValue)

  private class DelayedSchedule[V](
      underlying: LearningRateSchedule,
      delaySteps: Tensor0[Int32]
  ) extends LearningRateSchedule(underlying.steps + delaySteps):

    override def apply(step: Tensor0[Int32]): Tensor0[Float32] =
      val shiftedStep = maximum(step - delaySteps, Tensor0(0))
      underlying(shiftedStep)

  def apply(f: Tensor0[Int32] => Tensor0[Float32]): LearningRateSchedule =
    new LearningRateSchedule(steps = Tensor0(Int.MaxValue)):
      override def apply(step: Tensor0[Int32]): Tensor0[Float32] = f(step)

  extension (schedule: LearningRateSchedule)

    /** Shifts a schedule forward in time by a specified number of steps.
      *
      * For all `t < steps`, the schedule evaluates as if `t = 0`, effectively locking
      * the learning rate at its initial starting value until the delay has passed.
      *
      * @param delaySteps The number of iterations to delay the schedule's progression.
      * @return A time-shifted schedule.
      */
    def delay(delaySteps: Tensor0[Int32]): LearningRateSchedule = DelayedSchedule(schedule, delaySteps)

    def followBy(second: LearningRateSchedule): LearningRateSchedule = FollowBySchedule(schedule, second)

object LearningRateSchedules:

  /** Creates a schedule that maintains a constant learning rate for a specified number of steps. */
  class ConstantLearningRate(val learningRate: Tensor0[Float32], steps: Int = Int.MaxValue) extends LearningRateSchedule(steps = Tensor0(steps)):
    override def apply(step: Tensor0[Int32]): Tensor0[Float32] = learningRate

  /** Creates a schedule that rises linearly from `minLr` to `maxLr` over a specified number of warmup steps.
    *
    * @param vtype The floating-point type to use for the learning rate values.
    * @param from The initial learning rate at the start of the warmup (typically 0.0).
    * @param to The peak learning rate reached at the end of the warmup.
    * @param warmupSteps The number of steps over which the learning rate increases linearly.
    * @return A linear warmup schedule.
    */
  class LinearSchedule(
      val from: Tensor0[Float32],
      val to: Tensor0[Float32],
      val warmupSteps: Tensor0[Int32]
  ) extends LearningRateSchedule(steps = warmupSteps):

    private val vtype = from.vtype

    override def apply(step: Tensor0[Int32]): Tensor0[Float32] =
      val warmupRatio = minimum((step.asFloat(vtype)) / ((warmupSteps + 1).asFloat(vtype)), 1f)
      from + warmupRatio * (to - from)

  object LinearWarmup:
    def apply(
        to: Tensor0[Float32],
        warmupSteps: Tensor0[Int32]
    ): LinearSchedule = new LinearSchedule(0.0f, to, warmupSteps)

  /** Creates a schedule that decays from `maxLr` down to `minLr` following a half-cosine curve.
    *
    * This schedule has no concept of warmup; it begins decaying immediately at `t = 0`.
    * Once `t >= decaySteps`, the learning rate locks permanently at `minLr`.
    *
    * @param from The initial maximum learning rate at `t = 0`.
    * @param to The final baseline learning rate to reach after decaying.
    * @param decaySteps The number of steps over which to apply the decay curve.
    */
  class CosineDecay(
      val from: Tensor0[Float32],
      val to: Tensor0[Float32],
      val decaySteps: Tensor0[Int32]
  ) extends LearningRateSchedule(steps = decaySteps):

    private val vtype = from.vtype

    override def apply(step: Tensor0[Int32]): Tensor0[Float32] =
      val decayRatio = minimum((step - 1).asFloat(vtype) / decaySteps.asFloat(vtype), 1f)
      val coeff = 0.5f * (1.0f + (math.Pi.toFloat * decayRatio).cos)
      to + coeff * (from - to)

export LearningRateSchedules.*
