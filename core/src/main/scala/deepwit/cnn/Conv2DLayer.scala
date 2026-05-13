package deepwit.cnn

import dimwit.*
import deepwit.init

trait Conv2DLayer[S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V: IsFloating] extends (Tensor[S1 *: S2 *: InChannel *: EmptyTuple, V] => Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, V])

object Conv2DLayer:

  case class HyperParams[S1, S2](
      stride: Stride2[S1, S2] | Int = 1,
      padding: Padding = Padding.SAME
  )

  def xavierUniformKernel[S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], vtype: VType[V], key: Random.Key): Tensor[(S1, S2, InChannel, OutChannel), V] =
    val fanIn = s1Extent * s2Extent * channelExtent
    val fanOut = outChannelExtent
    val flatKernel = init.xavierUniform(fanIn, fanOut, vtype, key)
    flatKernel.unflatten(
      Axis[S1 |*| S2 |*| InChannel],
      Shape(s1Extent, s2Extent, channelExtent)
    )
