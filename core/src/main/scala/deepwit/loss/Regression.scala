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

  def apply[V: IsFloating](target: Tensor0[V], prediction: Tensor0[V], threshold: Float): Tensor0[V] =
    require(threshold > 0f, s"A transition point must be positive, but was $threshold.")
    val residual = AbsoluteError(target, prediction)
    val squared = 0.5f * SquaredError(target, prediction)
    val absolute = threshold * (residual - 0.5f * threshold)
    where(residual <= threshold, squared, absolute)
