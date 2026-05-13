package deepwit.base

import dimwit.*
import deepwit.init

case class AffineLayer[In: Label, Out: Label, V: IsFloating](params: AffineLayer.Params[In, Out, V]) extends (Tensor1[In, V] => Tensor1[Out, V]):

  override def apply(x: Tensor1[In, V]): Tensor1[Out, V] =
    x.dot(Axis[In])(params.weight) + params.bias

object AffineLayer:
  case class Params[In, Out, V](
      weight: Tensor2[In, Out, V],
      bias: Tensor1[Out, V]
  )

  object Params:

    def identity[In: Label, V: IsFloating](extent: AxisExtent[In], vtype: VType[V]): Params[In, Prime[In], V] =
      Params(
        weight = Tensor2.eye(extent, vtype),
        bias = Tensor(Shape(Axis[Prime[In]] -> extent.size), vtype).fill(0f)
      )

    def xavierNormal[In: Label, Out: Label, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, Out, V] =
      Params(
        weight = init.xavierNormal(inExtent, outExtent, vtype, key, gain = gain),
        bias = Tensor(Shape(outExtent), vtype).fill(0f)
      )

    def xavierUniform[In: Label, Out: Label, V: IsFloating](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], vtype: VType[V], key: Random.Key, gain: Float = 1f): Params[In, Out, V] =
      Params(
        weight = init.xavierUniform(inExtent, outExtent, vtype, key, gain = gain),
        bias = Tensor(Shape(outExtent), vtype).fill(0f)
      )
