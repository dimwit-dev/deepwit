package deepwit.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

import deepwit.{defaultEpsilon, unwrapEpsilon}

class RMSNorm[L: Λ, V: IsFloating](
    params: RMSNorm.Params[L, V],
    epsilon: Float | (DType => Float) = defaultEpsilon
) extends (Tensor1[L, V] => Tensor1[L, V]):

  private val ε: Tensor0[V] = Tensor0(VType[V])(unwrapEpsilon(epsilon, VType[V].dtype))

  def apply(x: Tensor1[L, V]): Tensor1[L, V] =
    def rescale(x: Tensor1[L, V]): Tensor1[L, V] =
      // Unlike LayerNorm, RMSNorm does not re-center: it only divides by the root mean square.
      val meanSquare = x.pow(2).mean
      x /! (meanSquare + ε).sqrt
    rescale(x) * params.weight

object RMSNorm:

  case class Params[L, V](weight: Tensor1[L, V])

  object Params:

    def init[L: Λ, V: IsFloating](ae: AxisExtent[L], vtype: VType[V] = VType[Float32]) = identity(ae, vtype)

    def identity[L: Λ, V: IsFloating](ae: AxisExtent[L], vtype: VType[V] = VType[Float32]) =
      Params(weight = Tensor(Shape(ae), vtype).fill(1f))
