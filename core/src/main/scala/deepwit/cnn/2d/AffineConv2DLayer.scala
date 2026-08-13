package deepwit.cnn.`2d`

import dimwit.*
import dimwit.Label as Λ

case class AffineConv2DLayer[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](
    params: AffineConv2DLayer.Params[S1, S2, InChannel, OutChannel, V],
    stride: Stride2[S1, S2] | Int = 1,
    padding: Padding = Padding.SAME
) extends (Tensor3[S1, S2, InChannel, V] => Tensor3[S1, S2, OutChannel, V]):

  override def apply(x: Tensor3[S1, S2, InChannel, V]): Tensor3[S1, S2, OutChannel, V] =
    x.conv2d(params.kernel, stride, padding) +! params.bias

object AffineConv2DLayer:

  case class Params[S1, S2, InChannel, OutChannel, V](
      kernel: Tensor[(S1, S2, InChannel, OutChannel), V],
      bias: Tensor1[OutChannel, V]
  )

  object Params:

    def xavierUniform[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], vtype: VType[V], key: Key): Params[S1, S2, InChannel, OutChannel, V] =
      Params(
        kernel = xavierUniformKernel(s1Extent, s2Extent, channelExtent, outChannelExtent, vtype, key),
        bias = Tensor(Shape(outChannelExtent), vtype).fill(0f)
      )
