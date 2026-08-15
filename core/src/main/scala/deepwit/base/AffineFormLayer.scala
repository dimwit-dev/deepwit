package deepwit.base

import dimwit.*
import dimwit.Label as Λ
import deepwit.init

/** Represents a learnable affine form.
  *
  * Mathematically, this layer computes $y = x \cdot w + b$, where $x$ is the
  * input tensor, $w$ is the learnable weight vector and $b$ is the learnable
  * scalar bias. It maps a vector to a scalar and is the rank-one special case
  * of an [[AffineLayer]]. For the bias-free equivalent, see [[LinearFormLayer]].
  *
  * @tparam In The axis label for the input dimension.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The layer parameters.
  */
class AffineFormLayer[In: Λ, V: IsFloating](params: AffineFormLayer.Params[In, V]) extends (Tensor1[In, V] => Tensor0[V]):
  override def apply(x: Tensor1[In, V]): Tensor0[V] =
    x.dot(Axis[In])(params.weight) + params.bias

object AffineFormLayer:

  /** Holds the learnable parameters for an [[AffineFormLayer]].
    *
    * @param weight The weight vector.
    * @param bias The scalar bias.
    */
  case class Params[In, V](
      weight: Tensor1[In, V],
      bias: Tensor0[V]
  )

  object Params:

    def xavierNormal[In: Λ, V: IsFloating](inExtent: AxisExtent[In], vtype: VType[V], key: Key, gain: Float = 1f): Params[In, V] =
      Params(
        weight = init.xavierNormalVector(inExtent, vtype, key, gain = gain),
        bias = Tensor0(vtype)(0f)
      )

    def xavierUniform[In: Λ, V: IsFloating](inExtent: AxisExtent[In], vtype: VType[V], key: Key, gain: Float = 1f): Params[In, V] =
      Params(
        weight = init.xavierUniformVector(inExtent, vtype, key, gain = gain),
        bias = Tensor0(vtype)(0f)
      )
