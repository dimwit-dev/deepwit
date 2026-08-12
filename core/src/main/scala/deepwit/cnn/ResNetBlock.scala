package deepwit.cnn

import dimwit.*
import deepwit.base.ResidualBlock
import deepwit.base.ActivationFunction.relu
import deepwit.cnn.Conv2DLayer.HyperParams

object ResNetBlock:
  type Params[H, W, C, V] = ResidualBlock.Params[AffineConv2DLayer.Params[H, W, C, C, V]]

  def apply[H: Label, W: Label, C: Label, V: IsFloating](
      params: Params[H, W, C, V]
  ): Tensor3[H, W, C, V] => Tensor3[H, W, C, V] = ResNetBlock(params, relu)

  def apply[H: Label, W: Label, C: Label, V: IsFloating](
      params: Params[H, W, C, V],
      activationF: Tensor[(H, W, C), V] => Tensor[(H, W, C), V]
  ): Tensor3[H, W, C, V] => Tensor3[H, W, C, V] =
    new ResidualBlock(
      params,
      p => AffineConv2DLayer(Conv2DLayer.HyperParams(stride = 1, padding = Padding.SAME))(p),
      activationF
    )

  object Params:

    def xavierUniform[H: Label, W: Label, C: Label, V: IsFloating](
        s1Extent: AxisExtent[H],
        s2Extent: AxisExtent[W],
        channelExtent: AxisExtent[C],
        vtype: VType[V],
        numLayers: Int,
        key: Random.Key
    ): Params[H, W, C, V] =
      require(numLayers >= 1, "At least one layer is required for a ResNetBlock.")
      ResidualBlock.Params.init(numLayers, key): (layerKey) =>
        AffineConv2DLayer.Params.xavierUniform(
          s1Extent,
          s2Extent,
          channelExtent,
          channelExtent,
          vtype,
          layerKey
        )
