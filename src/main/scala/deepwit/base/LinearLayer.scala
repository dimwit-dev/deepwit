package deepwit.base

import dimwit.*
import deepwit.init

case class LinearLayer[In: Label, Out: Label](params: LinearLayer.Params[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
  override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
    x.dot(Axis[In])(params.weight)

object LinearLayer:

  def apply[In: Label, Out: Label](weight: Tensor2[In, Out, Float]): LinearLayer[In, Out] =
    LinearLayer(LinearLayer.Params(weight))

  case class Params[In, Out](weight: Tensor2[In, Out, Float])

  object Params:

    def xavierNormal[In: Label, Out: Label](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key, gain: Float = 1f): Params[In, Out] =
      Params(weight = init.xavierNormal(inExtent, outExtent, key, gain = gain))

    def xavierUniform[In: Label, Out: Label](inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key, gain: Float = 1f): Params[In, Out] =
      Params(weight = init.xavierUniform(inExtent, outExtent, key, gain = gain))
