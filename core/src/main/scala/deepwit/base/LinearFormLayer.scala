package deepwit.base

import dimwit.*
import deepwit.init

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
case class LinearFormLayer[In: Label, V: IsFloating](params: LinearFormLayer.Params[In, V]) extends (Tensor1[In, V] => Tensor0[V]):
  override def apply(x: Tensor1[In, V]): Tensor0[V] =
    x.dot(Axis[In])(params.weight)

object LinearFormLayer:

  /** Creates a [[LinearFormLayer]] directly from a given weight vector.
    *
    * @param weight The weight vector.
    */
  def apply[In: Label, V: IsFloating](weight: Tensor1[In, V]): LinearFormLayer[In, V] =
    LinearFormLayer(LinearFormLayer.Params(weight))

  /** Holds the learnable parameters for a [[LinearFormLayer]].
    *
    * @param weight The weight vector.
    */
  case class Params[In, V](weight: Tensor1[In, V])

  object Params:

    def xavierNormal[In: Label, V: IsFloating](inExtent: AxisExtent[In], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, V] =
      Params(weight = init.xavierNormalVector(inExtent, vtype, key, gain = gain))

    def xavierUniform[In: Label, V: IsFloating](inExtent: AxisExtent[In], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, V] =
      Params(weight = init.xavierUniformVector(inExtent, vtype, key, gain = gain))
