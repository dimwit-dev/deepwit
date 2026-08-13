package deepwit.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.Label as Λ

import deepwit.{defaultEpsilon, unwrapEpsilon}

case class RMSNorm[L: Λ, V: IsFloating](
    params: RMSNorm.Params[L, V],
    epsilon: Float | (DType => Float) = defaultEpsilon
) extends (Tensor1[L, V] => Tensor1[L, V]):

  private val ε: Tensor0[V] = Tensor0(VType[V])(unwrapEpsilon(epsilon, VType[V].dtype))

  def apply(x: Tensor1[L, V]): Tensor1[L, V] =
    def rescale(x: Tensor1[L, V]): Tensor1[L, V] =
      val variance = (x -! x.mean).pow(2).mean
      x /! (variance + ε).sqrt
    rescale(x) * params.weight

object RMSNorm:

  case class Params[L, V](weight: Tensor1[L, V])

  object Params:

    def identity[L: Λ, V: IsFloating](ae: AxisExtent[L], vtype: VType[V]) =
      Params(weight = Tensor(Shape(ae), vtype).fill(1f))
