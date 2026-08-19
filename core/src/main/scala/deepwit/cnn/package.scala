package deepwit.cnn

import dimwit.*
import dimwit.Label as Λ

import deepwit.init.Init

/** The extent of a 2D pooling window, one extent per spatial axis. */
type Window2[S1, S2] = (AxisExtent[S1], AxisExtent[S2])

/** Samples a 2D convolution kernel from the Xavier/Glorot uniform distribution.
  *
  * The kernel is initialized as a flat matrix whose fan-in spans the spatial axes and the input
  * channels, and is then unflattened back into kernel shape.
  */
private[cnn] def xavierUniformKernel[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], key: Key, vtype: VType[V] = VType[Float32]): Tensor[(S1, S2, InChannel, OutChannel), V] =
  val fanIn = s1Extent * s2Extent * channelExtent
  val fanOut = outChannelExtent
  val flatKernel = Init.xavierUniform(fanIn, fanOut, key, vtype)
  flatKernel.unflatten(
    Axis[S1 |*| S2 |*| InChannel],
    Shape(s1Extent, s2Extent, channelExtent)
  )
