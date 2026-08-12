package deepwit.mlp

import dimwit.*
import deepwit.base.AffineLayer
import deepwit.base.ActivationFunction.relu
import deepwit.init
import deepwit.base.SequentialBlock

class MultiLayerPerceptron[L: Label, V: IsFloating](params: MultiLayerPerceptron.Params[L, V])
    extends SequentialBlock[AffineLayer.Params[L, L, V], Tensor1[L, V]](
      params,
      buildLayer = (params: AffineLayer.Params[L, L, V]) => new AffineLayer(params),
      activationF = relu
    )

object MultiLayerPerceptron:
  type Params[L, V] = SequentialBlock.Params[AffineLayer.Params[L, L, V]]

  object Params:

    def xavierNormal[L: Label, V: IsFloating](
        axis: Axis[L],
        vtype: VType[V],
        extents: List[Int],
        key: Random.Key,
        gain: Float = 1f
    ): Params[L, V] =
      val layerConfigs = extents.zip(extents.tail)
      SequentialBlock.Params.initFrom(layerConfigs, key): (config, layerKey) =>
        val (inExtent, outExtent) = config
        AffineLayer.Params.xavierNormal(
          inExtent = axis -> inExtent,
          outExtent = axis -> outExtent,
          vtype = vtype,
          key = layerKey,
          gain = gain
        )

    def xavierUniform[L: Label, V: IsFloating](
        axis: Axis[L],
        vtype: VType[V],
        extents: List[Int],
        key: Random.Key,
        gain: Float = 1f
    ): Params[L, V] =
      val layerConfigs = extents.zip(extents.tail)
      SequentialBlock.Params.initFrom(layerConfigs, key): (config, layerKey) =>
        val (inExtent, outExtent) = config
        AffineLayer.Params.xavierUniform(
          inExtent = axis -> inExtent,
          outExtent = axis -> outExtent,
          vtype = vtype,
          key = layerKey,
          gain = gain
        )
