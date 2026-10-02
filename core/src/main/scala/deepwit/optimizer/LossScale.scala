package deepwit.optimizer

import dimwit.*
import dimwit.TreeOf
import dimwit.TreeOf.mapLeaves
import dimwit.TreeOf.ops.**!

/** Dynamic loss scaling, for training in Float16 with Float32 master weights.
  *
  * Float16 rounds values below about 6e-8 to zero, and small gradients often are. Multiplying the loss by
  * `scale` multiplies every gradient by it as well, which keeps them in range. Too large a scale overflows
  * to inf instead, so a step whose gradients are not all finite is skipped, and [[next]] adapts the scale.
  *
  * BFloat16 has the range of Float32 and needs none of this.
  *
  * {{{
  * val grads = state.lossScale.unscaled(Autodiff.grad(p => state.lossScale.scaled(cost(p)))(state.params))
  * val (params, optState) = optimizer.update(grads, state.params, state.optState)
  * val finite = allFinite(grads)
  * TrainState(
  *   select(finite, params, state.params),
  *   select(finite, optState, state.optState),
  *   state.lossScale.next(finite)
  * )
  * }}}
  *
  * @param finiteSteps Steps in a row with finite gradients since the scale last changed.
  */
case class LossScale(scale: Tensor0[Float32], finiteSteps: Tensor0[Int32]):

  def scaled(loss: Tensor0[Float32]): Tensor0[Float32] = loss * scale

  def unscaled[H: TensorTree, V](scaledGrads: Grad[H])(using TreeOf[H, V])(using IsFloating[V]): Grad[H] =
    Grad(scaledGrads.value **! (1f / scale).asFloat(VType[V]))

  /** Halves the scale after an overflow, and doubles it after `growthInterval` finite steps in a row.
    *
    * @param minScale If the scale gets stuck here, the forward pass overflows, which no scale can fix.
    */
  def next(
      gradsFinite: Tensor0[Bool],
      growthInterval: Int = 2000,
      growthFactor: Float = 2f,
      backoffFactor: Float = 0.5f,
      minScale: Float = 1f
  ): LossScale =
    val counted = LossScale(scale, finiteSteps + 1)
    val grown = LossScale(scale * growthFactor, 0)
    val backedOff = LossScale(maximum(scale * backoffFactor, minScale), 0)
    val afterFiniteStep = select(counted.finiteSteps >= growthInterval, grown, counted)
    select(gradsFinite, afterFiniteStep, backedOff)

object LossScale:
  def initial(scale: Float = 65536f): LossScale = LossScale(Tensor0(scale), Tensor0(0))

def allFinite[H: TensorTree, V](grads: Grad[H])(using TreeOf[H, V])(using IsFloating[V]): Tensor0[Bool] =
  grads.value.mapLeaves([T <: Tuple] => (labels: Labels[T]) ?=> (grad: Tensor[T, V]) => grad.isfinite.all).reduce(_ and _)

/** `ifTrue` if `condition` holds, else `ifFalse`, leaf by leaf. Unlike an `if`, this works inside `jit`. */
def select[P: TensorTree](condition: Tensor0[Bool], ifTrue: P, ifFalse: P): P =
  TensorTree[P].zipMap(ifTrue, ifFalse, [T <: Tuple, V] => (labels: Labels[T]) ?=> (a: Tensor[T, V], b: Tensor[T, V]) => where_!(condition, a, b))
