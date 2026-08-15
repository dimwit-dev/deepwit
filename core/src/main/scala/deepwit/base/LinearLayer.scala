package deepwit.base

import dimwit.*
import deepwit.init
import dimwit.Label as Λ

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
class LinearLayer[In: Λ, Out: Λ, V: IsFloating](params: LinearLayer.Params[In, Out, V]) extends (Tensor1[In, V] => Tensor1[Out, V]):
  override def apply(x: Tensor1[In, V]): Tensor1[Out, V] =
    x.dot(Axis[In])(params.weight)

object LinearLayer:

  def apply[In: Λ, Out: Λ, V: IsFloating](params: LinearLayer.Params[In, Out, V]): LinearLayer[In, Out, V] =
    new LinearLayer(params)

  /** Creates a [[LinearLayer]] directly from a given weight matrix.
    *
    * @param weight The weight matrix.
    */
  def apply[In: Λ, Out: Λ, V: IsFloating](weight: Tensor2[In, Out, V]): LinearLayer[In, Out, V] =
    new LinearLayer(LinearLayer.Params(weight))

  /** Holds the learnable parameters for a [[LinearLayer]].
    *
    * @param weight The weight matrix.
    */
  case class Params[In, Out, V](weight: Tensor2[In, Out, V])

  object Params:

    def identity[In: Λ, V: IsFloating](extent: AxisExtent[In], vtype: VType[V]): Params[In, Prime[In], V] = Params(weight = Tensor2.eye(extent, vtype))

    def xavierNormal[In: Λ, Out: Λ, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, Out, V] =
      Params(weight = init.xavierNormal(inExtent, outExtent, vtype, key, gain = gain))

    def xavierUniform[In: Λ, Out: Λ, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, Out, V] =
      Params(weight = init.xavierUniform(inExtent, outExtent, vtype, key, gain = gain))
