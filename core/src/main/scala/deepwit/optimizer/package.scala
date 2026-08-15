package deepwit.optimizer

import dimwit.*
import dimwit.Conversions.given
import dimwit.TreeOf
import dimwit.TreeOf.{map, mapLeaves}

extension [H: TensorTree, V: IsFloating](grads: Grad[H])(using TreeOf[H, V])

  /** Rescales all gradients by a common factor so that their global L2 norm does not exceed `maxNorm`. */
  def clipGlobalNorm(maxNorm: Tensor0[V], epsilon: Double = 1e-6): Grad[H] =
    val gradNorm = grads.value.mapLeaves([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x.pow(2).sum).reduce(_ + _).sqrt
    val scale = minimum(1f, maxNorm / (gradNorm + epsilon))
    Grad(grads.value.map([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x *! scale))
