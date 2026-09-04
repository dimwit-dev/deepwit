package deepwit.optimizer

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.GradientOptimizer

case class LearningRateSchedulerState[P, State[_]](
    step: Tensor0[Int32],
    optState: State[P]
)
type LearningRateSchedulerStateFor[State[_]] = [P] =>> LearningRateSchedulerState[P, State]

/** Wraps an optimizer so its learning rate follows `schedule`, counting steps from 1.
  *
  * @param optF Builds the optimizer at a given learning rate, e.g. `lr => Adam(lr)`.
  */
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

/** A learning rate as a function of the step count, which starts at 1.
  *
  * @param steps How many steps this schedule is defined for, after which it holds its final value.
  *              [[LearningRateSchedule.followBy]] uses it to know when to hand over.
  */
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
      */
    def delay(delaySteps: Tensor0[Int32]): LearningRateSchedule = DelayedSchedule(schedule, delaySteps)

    /** Runs this schedule through its `steps`, then hands over to `second` delayed by that much,
      * so `second` starts from its own beginning rather than mid-curve.
      */
    def followBy(second: LearningRateSchedule): LearningRateSchedule = FollowBySchedule(schedule, second)

object LearningRateSchedules:

  /** A schedule of constant `learningRate` for a specified number of `steps`. */
  class ConstantLearningRate(val learningRate: Tensor0[Float32], steps: Int = Int.MaxValue) extends LearningRateSchedule(steps = Tensor0(steps)):
    override def apply(step: Tensor0[Int32]): Tensor0[Float32] = learningRate

  /** A schedule rising linearly from `from` to `to` for a specified number of `warmupSteps`. */
  class LinearSchedule(
      val from: Tensor0[Float32],
      val to: Tensor0[Float32],
      val warmupSteps: Tensor0[Int32]
  ) extends LearningRateSchedule(steps = warmupSteps):

    private val vtype = from.vtype

    override def apply(step: Tensor0[Int32]): Tensor0[Float32] =
      // Steps are 1-based, and the +1 keeps the first step off `from` — with LinearWarmup's
      // `from = 0` that would be a step at zero learning rate.
      val warmupRatio = minimum((step.asFloat(vtype)) / ((warmupSteps + 1).asFloat(vtype)), 1f)
      from + warmupRatio * (to - from)

  object LinearWarmup:
    def apply(
        to: Tensor0[Float32],
        warmupSteps: Tensor0[Int32]
    ): LinearSchedule = new LinearSchedule(0.0f, to, warmupSteps)

  /** A schedule that decays from `from` down to `to` for `decaySteps` following a half-cosine curve.
    *
    * Past `decaySteps` the learning rate stays at `to`.
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
