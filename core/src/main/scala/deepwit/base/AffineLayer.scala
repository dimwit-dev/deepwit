package deepwit.base

import dimwit.*
import dimwit.Label as Λ

/** Represents a learnable affine transformation.
  *
  * Mathematically, this layer computes $y = x W + b$, where $x$ is the input
  * tensor, $W$ is the learnable weight matrix, and $b$ is the learnable bias
  * vector. It is commonly referred to as a fully connected or dense layer.
  * For the bias-free equivalent, see [[LinearLayer]].
  *
  * @tparam In The axis label for the input dimension.
  * @tparam Out The axis label for the output dimension.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The layer parameters.
  */
class AffineLayer[In: Λ, Out: Λ, V: IsFloating](params: AffineLayer.Params[In, Out, V]) extends (Tensor1[In, V] => Tensor1[Out, V]):

  override def apply(x: Tensor1[In, V]): Tensor1[Out, V] =
    x.dot(Axis[In])(params.weight) + params.bias

object AffineLayer:

  /** Holds the learnable parameters for an [[AffineLayer]].
    *
    * @param weight The weight matrix.
    * @param bias The bias vector.
    */
  case class Params[In, Out, V](
      weight: Tensor2[In, Out, V],
      bias: Tensor1[Out, V]
  )

  object Params:

    def identity[In: Λ, V: IsFloating](extent: AxisExtent[In], vtype: VType[V]): Params[In, Prime[In], V] =
      Params(
        weight = Tensor2.eye(extent, vtype),
        bias = Tensor(Shape(Axis[Prime[In]] -> extent.size), vtype).fill(0f)
      )

    def xavierNormal[In: Λ, Out: Λ, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Key, gain: Float = 1f): Params[In, Out, V] =
      Params(
        weight = deepwit.init.xavierNormal(inExtent, outExtent, vtype, key, gain = gain),
        bias = Tensor(Shape(outExtent), vtype).fill(0f)
      )

    def xavierUniform[In: Λ, Out: Λ, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Key, gain: Float = 1f): Params[In, Out, V] =
      Params(
        weight = deepwit.init.xavierUniform(inExtent, outExtent, vtype, key, gain = gain),
        bias = Tensor(Shape(outExtent), vtype).fill(0f)
      )
