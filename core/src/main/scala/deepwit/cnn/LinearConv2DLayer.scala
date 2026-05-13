package deepwit.cnn

import dimwit.*
import dimwit.random.Random.Key
import deepwit.init

case class LinearConv2DLayer[S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V: IsFloating](
    hyperParams: Conv2DLayer.HyperParams[S1, S2]
)(
    params: LinearConv2DLayer.Params[S1, S2, InChannel, OutChannel, V]
) extends Conv2DLayer[S1, S2, InChannel, OutChannel, V]:

  override def apply(x: Tensor[S1 *: S2 *: InChannel *: EmptyTuple, V]): Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, V] =
    x.conv2d(params.kernel, hyperParams.stride, hyperParams.padding)

object LinearConv2DLayer:

  export Conv2DLayer.HyperParams

  case class Params[S1, S2, InChannel, OutChannel, V: IsFloating](
      kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, V]
  )

  object Params:

    def xavierUniform[S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], vtype: VType[V], key: Random.Key): Params[S1, S2, InChannel, OutChannel, V] =
      Params(kernel = Conv2DLayer.xavierUniformKernel(s1Extent, s2Extent, channelExtent, outChannelExtent, vtype, key))
