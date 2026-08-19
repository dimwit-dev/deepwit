package deepwit.base

import dimwit.*
import deepwit.init.Init
import dimwit.Label as Λ

/** Represents a learnable linear form.
  *
  * Mathematically, this layer computes $y = x \cdot w$, where $x$ is the input
  * tensor and $w$ is the learnable weight vector. It maps a vector to a scalar
  * and is the rank-one special case of a [[LinearLayer]]. For the variant with
  * a bias term, see [[AffineFormLayer]].
  *
  * @tparam In The axis label for the input dimension.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The layer parameters.
  */
class LinearFormLayer[In: Λ, V: IsFloating](params: LinearFormLayer.Params[In, V]) extends (Tensor1[In, V] => Tensor0[V]):
  override def apply(x: Tensor1[In, V]): Tensor0[V] =
    x.dot(Axis[In])(params.weight)

object LinearFormLayer:

  def apply[In: Λ, V: IsFloating](params: LinearFormLayer.Params[In, V]): LinearFormLayer[In, V] =
    new LinearFormLayer(params)

  /** Creates a [[LinearFormLayer]] directly from a given weight vector.
    *
    * @param weight The weight vector.
    */
  def apply[In: Λ, V: IsFloating](weight: Tensor1[In, V]): LinearFormLayer[In, V] =
    new LinearFormLayer(LinearFormLayer.Params(weight))

  /** Holds the learnable parameters for a [[LinearFormLayer]].
    *
    * @param weight The weight vector.
    */
  case class Params[In, V](weight: Tensor1[In, V])

  object Params:

    def init[In: Λ, V: IsFloating](inExtent: AxisExtent[In], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1f): Params[In, V] =
      xavierUniform(inExtent, key, vtype, gain)

    def xavierNormal[In: Λ, V: IsFloating](inExtent: AxisExtent[In], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1f): Params[In, V] =
      Params(weight = Init.xavierNormalVector(inExtent, key, vtype, gain = gain))

    def xavierUniform[In: Λ, V: IsFloating](inExtent: AxisExtent[In], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1f): Params[In, V] =
      Params(weight = Init.xavierUniformVector(inExtent, key, vtype, gain = gain))
