package deepwit.cnn

import dimwit.*
import dimwit.random.Random.Key
import deepwit.init

case class LinearConv2DLayer[S1: Label, S2: Label, InChannel: Label, OutChannel: Label](
    hyperParams: Conv2DLayer.HyperParams[S1, S2]
)(
    params: LinearConv2DLayer.Params[S1, S2, InChannel, OutChannel]
) extends Conv2DLayer[S1, S2, InChannel, OutChannel]:

  override def apply(x: Tensor[S1 *: S2 *: InChannel *: EmptyTuple, Float]): Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, Float] =
    x.conv2d(params.kernel, hyperParams.stride, hyperParams.padding)

object LinearConv2DLayer:

  export Conv2DLayer.HyperParams

  case class Params[S1, S2, InChannel, OutChannel](
      kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, Float]
  )

  object Params:

    def xavierUniform[S1: Label, S2: Label, InChannel: Label, OutChannel: Label](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], key: Random.Key): Params[S1, S2, InChannel, OutChannel] =
      Params(kernel = Conv2DLayer.xavierUniformKernel(s1Extent, s2Extent, channelExtent, outChannelExtent, key))
