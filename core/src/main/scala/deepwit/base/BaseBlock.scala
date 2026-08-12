package deepwit.base

import dimwit.*

class SequentialBlock[P, T](
    params: SequentialBlock.Params[P],
    buildLayer: P => (T => T),
    activationF: T => T
) extends (T => T):

  private val layers = params.layerParams.map(buildLayer)

  def apply(input: T): T =
    layers.foldLeft(input): (x, layer) =>
      activationF(layer(x))

object SequentialBlock:
  case class Params[P](layerParams: List[P])

  object Params:
    def initFrom[Config, P](
        configs: List[Config],
        key: Random.Key
    )(buildParam: (Config, Random.Key) => P): Params[P] =
      val layerKeys = key.split(configs.size)
      val layerParams = configs.zip(layerKeys).map:
        case (cfg, k) => buildParam(cfg, k)
      Params(layerParams)

class ResidualBlock[P, T <: Tuple: Labels, V: IsNumber](
    params: ResidualBlock.Params[P],
    buildLayer: P => (Tensor[T, V] => Tensor[T, V]),
    activationF: Tensor[T, V] => Tensor[T, V]
) extends (Tensor[T, V] => Tensor[T, V]):

  private val layers = params.layerParams.map(buildLayer)

  def apply(input: Tensor[T, V]): Tensor[T, V] =
    layers.foldLeft(input): (x, layer) =>
      activationF(x + layer(x))

object ResidualBlock:
  case class Params[P](layerParams: List[P])

  object Params:

    def init[P](numLayers: Int, key: Random.Key)(buildParam: Random.Key => P): Params[P] =
      val emptyConfigs = List.fill(numLayers)(())
      initFrom(emptyConfigs, key):
        case (_, layerKey) => buildParam(layerKey)

    def initFrom[Config, P](
        configs: List[Config],
        key: Random.Key
    )(buildParam: (Config, Random.Key) => P): Params[P] =
      val layerKeys = key.split(configs.size)
      val layerParams = configs.zip(layerKeys).map:
        case (cfg, k) => buildParam(cfg, k)
      Params(layerParams)
