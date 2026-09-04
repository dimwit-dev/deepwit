package deepwit.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

import deepwit.{defaultEpsilon, unwrapEpsilon}

/** Rescales by the root mean square over the `L` axis, then scales by the learned weight, as
  * described in [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
  *
  * @param epsilon Guards the division. Defaults to the machine epsilon of data type; pass a `Float` to fix it. Pass a function to derive it from the data type.
  */
class RMSNorm[L: Λ, V: IsFloating](
    params: RMSNorm.Params[L, V],
    epsilon: Float | (DType => Float) = defaultEpsilon
) extends (Tensor1[L, V] => Tensor1[L, V]):

  private val ε: Tensor0[V] = Tensor0(VType[V])(unwrapEpsilon(epsilon, VType[V].dtype))

  def apply(x: Tensor1[L, V]): Tensor1[L, V] =
    def rescale(x: Tensor1[L, V]): Tensor1[L, V] =
      val meanSquare = x.pow(2).mean
      x /! (meanSquare + ε).sqrt
    rescale(x) * params.weight

object RMSNorm:

  case class Params[L, V](weight: Tensor1[L, V])

  object Params:

    def init[L: Λ, V: IsFloating](ae: AxisExtent[L], vtype: VType[V] = VType[Float32]) = identity(ae, vtype)

    def identity[L: Λ, V: IsFloating](ae: AxisExtent[L], vtype: VType[V] = VType[Float32]) =
      Params(weight = Tensor(Shape(ae), vtype).fill(1f))
