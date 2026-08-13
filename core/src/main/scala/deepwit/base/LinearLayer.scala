package deepwit.base

import dimwit.*
import deepwit.init

/** Represents a learnable linear transformation.
  *
  * Mathematically, this layer computes $y = x W$, where $x$ is the input
  * tensor and $W$ is the learnable weight matrix. Unlike an [[AffineLayer]],
  * it performs a strictly linear mapping without a bias term.
  *
  * @tparam In The axis label for the input dimension.
  * @tparam Out The axis label for the output dimension.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The layer parameters.
  */
case class LinearLayer[In: Label, Out: Label, V: IsFloating](params: LinearLayer.Params[In, Out, V]) extends (Tensor1[In, V] => Tensor1[Out, V]):
  override def apply(x: Tensor1[In, V]): Tensor1[Out, V] =
    x.dot(Axis[In])(params.weight)

object LinearLayer:

  /** Creates a [[LinearLayer]] directly from a given weight matrix.
    *
    * @param weight The weight matrix.
    */
  def apply[In: Label, Out: Label, V: IsFloating](weight: Tensor2[In, Out, V]): LinearLayer[In, Out, V] =
    LinearLayer(LinearLayer.Params(weight))

  /** Holds the learnable weight for a [[LinearLayer]].
    *
    * @param weight The weight matrix.
    */
  case class Params[In, Out, V](weight: Tensor2[In, Out, V])

  object Params:

    def identity[In: Label, V: IsFloating](extent: AxisExtent[In], vtype: VType[V]): Params[In, Prime[In], V] = Params(weight = Tensor2.eye(extent, vtype))

    def xavierNormal[In: Label, Out: Label, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, Out, V] =
      Params(weight = init.xavierNormal(inExtent, outExtent, vtype, key, gain = gain))

    def xavierUniform[In: Label, Out: Label, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, Out, V] =
      Params(weight = init.xavierUniform(inExtent, outExtent, vtype, key, gain = gain))
