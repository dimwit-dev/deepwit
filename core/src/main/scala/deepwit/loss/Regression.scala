package deepwit.loss

import dimwit.*
import dimwit.Conversions.given

object SquaredError:

  def apply[V: IsFloating](target: Tensor0[V], prediction: Tensor0[V]): Tensor0[V] =
    (target - prediction).pow(2)

object AbsoluteError:

  def apply[V: IsFloating](target: Tensor0[V], prediction: Tensor0[V]): Tensor0[V] =
    (target - prediction).abs

object Huber:

  /** @param transitionPoint Where the quadratic branch hands over to the linear one. Must be
    *                        positive. A scalar rather than a `Float` so it can be traced, and
    *                        scheduled the way a learning rate is.
    */
  def apply[V: IsFloating](target: Tensor0[V], prediction: Tensor0[V], transitionPoint: Tensor0[Float32]): Tensor0[V] =
    val δ = transitionPoint.asFloat(VType[V])
    val residual = AbsoluteError(target, prediction)
    val squared = 0.5f * SquaredError(target, prediction)
    val absolute = δ * (residual - 0.5f * δ)
    where(residual <= δ, squared, absolute)
