package deepwit.base

import dimwit.*
import deepwit.init

case class AffineLayer[In: Label, Out: Label](params: AffineLayer.Params[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):

  override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
    x.dot(Axis[In])(params.weight) + params.bias

object AffineLayer:
  case class Params[In, Out](
      weight: Tensor2[In, Out, Float],
      bias: Tensor1[Out, Float]
  )

  object Params:

    def xavierNormal[In: Label, Out: Label](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key, gain: Float = 1f): Params[In, Out] =
      Params(
        weight = init.xavierNormal(inExtent, outExtent, key, gain = gain),
        bias = Tensor(Shape(outExtent)).fill(0f)
      )

    def xavierUniform[In: Label, Out: Label](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key, gain: Float = 1f): Params[In, Out] =
      Params(
        weight = init.xavierUniform(inExtent, outExtent, key, gain = gain),
        bias = Tensor(Shape(outExtent)).fill(0f)
      )
