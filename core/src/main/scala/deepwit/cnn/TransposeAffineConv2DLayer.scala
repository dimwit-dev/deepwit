package deepwit.cnn

import dimwit.*
import dimwit.Label as Λ

/** The adjoint of a 2D convolution, with bias. For the bias-free equivalent, see
  * [[TransposeLinearConv2DLayer]].
  *
  * @tparam InChannel The forward convolution's input channels, so this layer's output.
  * @tparam OutChannel The forward convolution's output channels, so this layer's input.
  * @param stride An `Int` stride applies to both spatial axes. A stride above 1 grows the spatial extents, which is how these layers upsample.
  * @param padding `Padding.SAME` preserves the spatial extents, while `Padding.VALID` grows them.
  */
class TransposeAffineConv2DLayer[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](
    params: TransposeAffineConv2DLayer.Params[S1, S2, InChannel, OutChannel, V],
    stride: Stride2[S1, S2] | Int = 1,
    padding: Padding = Padding.SAME
) extends (Tensor3[S1, S2, OutChannel, V] => Tensor3[S1, S2, InChannel, V]):

  override def apply(x: Tensor3[S1, S2, OutChannel, V]): Tensor3[S1, S2, InChannel, V] =
    x.transposeConv2d(params.kernel, stride, padding) +! params.bias

object TransposeAffineConv2DLayer:

  case class Params[S1, S2, InChannel, OutChannel, V](
      kernel: Tensor[(S1, S2, InChannel, OutChannel), V],
      bias: Tensor1[InChannel, V]
  )

  object Params:

    def init[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], key: Key, vtype: VType[V] = VType[Float32]): Params[S1, S2, InChannel, OutChannel, V] =
      xavierUniform(s1Extent, s2Extent, channelExtent, outChannelExtent, key, vtype)

    def xavierUniform[S1: Λ, S2: Λ, InChannel: Λ, OutChannel: Λ, V: IsFloating](s1Extent: AxisExtent[S1], s2Extent: AxisExtent[S2], channelExtent: AxisExtent[InChannel], outChannelExtent: AxisExtent[OutChannel], key: Key, vtype: VType[V] = VType[Float32]): Params[S1, S2, InChannel, OutChannel, V] =
      Params(
        kernel = xavierUniformKernel(s1Extent, s2Extent, channelExtent, outChannelExtent, key, vtype),
        bias = Tensor(Shape(channelExtent), vtype).fill(0f)
      )
