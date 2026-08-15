package deepwit.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

import deepwit.{defaultEpsilon, unwrapEpsilon}


class LayerNorm[L: Λ, V: IsFloating](
    params: LayerNorm.Params[L, V],
    epsilon: Float | (DType => Float) = defaultEpsilon
) extends (Tensor1[L, V] => Tensor1[L, V]):

  private val ε: Tensor0[V] = Tensor0(VType[V])(unwrapEpsilon(epsilon, VType[V].dtype))

  def apply(x: Tensor1[L, V]): Tensor1[L, V] =
    def standardize(x: Tensor1[L, V]): Tensor1[L, V] =
      val x0 = x -! x.mean
      val variance = x0.pow(2).mean
      x0 /! (variance + ε).sqrt
    standardize(x) * params.weight + params.bias

object LayerNorm:

  case class Params[L, V](weight: Tensor1[L, V], bias: Tensor1[L, V])

  object Params:
    def identity[L: Λ, V: IsFloating](ae: AxisExtent[L], vtype: VType[V]) =
      Params(
        weight = Tensor(Shape(ae), vtype).fill(1f),
        bias = Tensor(Shape(ae), vtype).fill(0f)
      )
