package deepwit.cnn

import dimwit.*

package object `2d`:

  private def xavierUniformKernel[S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], vtype: VType[V], key: Random.Key): Tensor[(S1, S2, InChannel, OutChannel), V] =
    val fanIn = s1Extent * s2Extent * channelExtent
    val fanOut = outChannelExtent
    val flatKernel = deepwit.init.xavierUniform(fanIn, fanOut, vtype, key)
    flatKernel.unflatten(
      Axis[S1 |*| S2 |*| InChannel],
      Shape(s1Extent, s2Extent, channelExtent)
    )

  type Window2[S1, S2] = (AxisExtent[S1], AxisExtent[S2])
