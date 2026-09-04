package deepwit.cnn

import dimwit.*
import dimwit.Label as Λ

/** A 2D convolution without bias. For the bias-carrying equivalent, see [[AffineConv2DLayer]].
  *
  * @params stride An `Int` stride applies to both spatial axes.
  * @params padding `Padding.SAME` preserves the spatial extents, while `Padding.VALID` shrinks them.
  */
class LinearConv2DLayer[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](
    params: LinearConv2DLayer.Params[S1, S2, InChannel, OutChannel, V],
    stride: Stride2[S1, S2] | Int = 1,
    padding: Padding = Padding.SAME
) extends (Tensor3[S1, S2, InChannel, V] => Tensor3[S1, S2, OutChannel, V]):

  override def apply(x: Tensor3[S1, S2, InChannel, V]): Tensor3[S1, S2, OutChannel, V] =
    x.conv2d(params.kernel, stride, padding)

object LinearConv2DLayer:

  case class Params[S1, S2, InChannel, OutChannel, V](
      kernel: Tensor[(S1, S2, InChannel, OutChannel), V]
  )

  object Params:

    def init[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], key: Key, vtype: VType[V] = VType[Float32]): Params[S1, S2, InChannel, OutChannel, V] =
      xavierUniform(s1Extent, s2Extent, channelExtent, outChannelExtent, key, vtype)

    def xavierUniform[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], key: Key, vtype: VType[V] = VType[Float32]): Params[S1, S2, InChannel, OutChannel, V] =
      Params(kernel = xavierUniformKernel(s1Extent, s2Extent, channelExtent, outChannelExtent, key, vtype))
