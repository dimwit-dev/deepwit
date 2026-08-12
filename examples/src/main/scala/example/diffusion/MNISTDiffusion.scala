package deepwit.example.diffusion

import dimwit.*
import dimwit.Conversions.given

import deepwit.*

import examples.dataset.MNISTLoader.{Height, Width}
import java.sql.Time
import deepwit.mlp.MultiLayerPerceptron
import deepwit.cnn.ResNetBlock

trait Channel derives Label
trait Embedding derives Label

object TimeEmbedder extends (Tensor0[Float32] => Tensor1[Embedding, Float32]):

  private val MaxTimesteps = 1000f
  private val BaseFrequency = 10000f

  def apply(t: Tensor0[Float32]): Tensor1[Embedding, Float32] =
    val freqs = Tensor1(Axis[Embedding] -> 16, VType[Float32]).fromArray(Array.tabulate(16)(i => 1f / (math.pow(BaseFrequency, (i.toFloat / 16f)))))
    val scaledT = t * MaxTimesteps
    val angles = scaledT *! freqs
    concatenate(angles.sin, angles.cos, Axis[Embedding])

class DiffusionUNet(hyperParams: DiffusionUNet.HyperParams)(params: DiffusionUNet.Params) extends ((Tensor2[Height, Width, Float32], Tensor0[Float32]) => Tensor2[Height, Width, Float32]):

  private val stem = AffineConv2DLayer(hyperParams.stem)(params.stem)

  private val encoderLayers = params.encoders.map:
    case (resNetBlockParams, downLayerParams) =>
      val resNetBlock = ResNetBlock(resNetBlockParams, relu)
      val downLayer = AffineConv2DLayer(hyperParams.encoder)(downLayerParams)
      (resNetBlock andThen downLayer)

  private val bottleneckSpatial = AffineConv2DLayer(hyperParams.bottleneckSpatial)(params.bottleneckSpatial)

  private val decoderLayers = params.decoders.map:
    case (resNetBlockParams, upLayerParams) =>
      val resNetBlock = ResNetBlock(resNetBlockParams, relu)
      val upLayer = TransposeAffineConv2DLayer(hyperParams.decoder)(upLayerParams)
      (resNetBlock andThen upLayer)

  private val outputHead = AffineLayer(params.outputHead)

  private val timeEncoder = MultiLayerPerceptron(params.timeEncoder)

  private val bottleneckTimeProj = AffineLayer(params.bottleneckTimeProj)

  private val decodersTimeProj = params.decodersTimeProj.map(p => new AffineLayer(p))

  private val encodersTimeProj = params.encodersTimeProj.map(p => new AffineLayer(p))

  def apply(image: Tensor2[Height, Width, Float32], t: Tensor0[Float32]): Tensor2[Height, Width, Float32] =

    val timeEmbedding = timeEncoder(TimeEmbedder(t))

    val stemOut = relu(stem(image.appendAxis(Axis[Channel])))
    val encoderOuts = encoderLayers.zip(encodersTimeProj).scanLeft(stemOut):
      case (x, (layer, encoderTimeProj)) =>
        val layerTimeEmbedding = encoderTimeProj(timeEmbedding)
        val h = layer(x)
        relu(h +! layerTimeEmbedding)
    val (encoderOut, skips) = (encoderOuts.last, encoderOuts.reverse)

    val bottomSpatial = relu(bottleneckSpatial(encoderOut))
    val bottleneckTime = bottleneckTimeProj(timeEmbedding)
    val bottom = relu(bottomSpatial +! bottleneckTime)

    val decoderOut = decoderLayers.zip(decodersTimeProj).zip(skips).foldLeft(bottom):
      case (x, ((layer, decoderTimeProj), spatialSkip)) =>
        val layerTimeEmbedding = decoderTimeProj(timeEmbedding)
        val h = layer(concatenate(x, spatialSkip, Axis[Embedding]))
        relu(h +! layerTimeEmbedding)

    concatenate(decoderOut, stemOut, Axis[Embedding])
      .vmap(Axis[Height])(_.vmap(Axis[Width])(outputHead))
      .squeeze(Axis[Embedding])

object DiffusionUNet:
  case class HyperParams(
      stem: AffineConv2DLayer.HyperParams[Height, Width],
      encoder: Conv2DLayer.HyperParams[Height, Width],
      decoder: Conv2DLayer.HyperParams[Height, Width],
      bottleneckSpatial: Conv2DLayer.HyperParams[Height, Width]
  )

  object HyperParams:
    def default: HyperParams =
      HyperParams(
        stem = Conv2DLayer.HyperParams(stride = 1, padding = Padding.SAME),
        encoder = Conv2DLayer.HyperParams(stride = 2, padding = Padding.SAME),
        decoder = Conv2DLayer.HyperParams(stride = 2, padding = Padding.SAME),
        bottleneckSpatial = Conv2DLayer.HyperParams(stride = 1, padding = Padding.SAME)
      )

  case class Params(
      stem: AffineConv2DLayer.Params[Height, Width, Channel, Embedding, Float32],
      encoders: List[(resNetBlock: ResNetBlock.Params[Height, Width, Embedding, Float32], down: AffineConv2DLayer.Params[Height, Width, Embedding, Embedding, Float32])],
      bottleneckSpatial: AffineConv2DLayer.Params[Height, Width, Embedding, Embedding, Float32],
      decoders: List[(resNetBlock: ResNetBlock.Params[Height, Width, Embedding, Float32], up: TransposeAffineConv2DLayer.Params[Height, Width, Embedding, Embedding, Float32])],
      outputHead: AffineLayer.Params[Embedding, Embedding, Float32],
      timeEncoder: MultiLayerPerceptron.Params[Embedding, Float32],
      bottleneckTimeProj: AffineLayer.Params[Embedding, Embedding, Float32],
      encodersTimeProj: List[AffineLayer.Params[Embedding, Embedding, Float32]],
      decodersTimeProj: List[AffineLayer.Params[Embedding, Embedding, Float32]]
  )

  object Params:
    def xavierUniform(
        encoderExtents: List[Int],
        timeEncoderExtends: List[Int] = List(32, 32)
    )(key: Random.Key): Params =
      val (stemKey, encoderKey, bottleneckSpatialKey, decoderKey, headKey, timeEncoderKey, bottleneckTimeKey, encoderTimeKey, decoderTimeKey) = key.splitToTuple(9)
      val stemParams = AffineConv2DLayer.Params.xavierUniform(
        Axis[Height] -> 7,
        Axis[Width] -> 7,
        Axis[Channel] -> 1,
        Axis[Embedding] -> encoderExtents.head,
        VType[Float32],
        stemKey
      )
      val encoderParams = encoderExtents.zip(encoderExtents.tail).zip(encoderKey.split(encoderExtents.size)).map:
        case ((in, out), key) =>
          val (resNetBlockKey, downLayerKey) = key.splitToTuple(2)
          val resNetBlock = ResNetBlock.Params.xavierUniform(
            Axis[Height] -> 3,
            Axis[Width] -> 3,
            Axis[Embedding] -> in,
            VType[Float32],
            numLayers = 1,
            resNetBlockKey
          )
          val downLayer = AffineConv2DLayer.Params.xavierUniform(
            Axis[Height] -> 3,
            Axis[Width] -> 3,
            Axis[Embedding] -> in,
            Axis[Embedding] -> out,
            VType[Float32],
            downLayerKey
          )
          (resNetBlock, downLayer)
      val bottleneckSpatialParams = AffineConv2DLayer.Params.xavierUniform(
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[Embedding] -> encoderExtents.last,
        Axis[Embedding] -> encoderExtents.last,
        VType[Float32],
        bottleneckSpatialKey
      )
      val decoderExtents = encoderExtents.reverse
      val decoderParams = decoderExtents.zip(decoderExtents.tail).zip(decoderKey.split(decoderExtents.size)).map:
        case ((in, out), key) =>
          val (resNetBlockKey, upLayerKey) = key.splitToTuple(2)
          val resNetBlock = ResNetBlock.Params.xavierUniform(
            Axis[Height] -> 3,
            Axis[Width] -> 3,
            Axis[Embedding] -> in * 2, // input + skip connection
            VType[Float32],
            numLayers = 1,
            resNetBlockKey
          )
          val upLayer = TransposeAffineConv2DLayer.Params.xavierUniform(
            Axis[Height] -> 3,
            Axis[Width] -> 3,
            Axis[Embedding] -> out,
            Axis[Embedding] -> in * 2, // input + skip connection
            VType[Float32],
            upLayerKey
          )
          (resNetBlock, upLayer)
      val outputHeadParams = AffineLayer.Params.xavierUniform(
        Axis[Embedding] -> encoderExtents.head * 2,
        Axis[Embedding] -> 1,
        VType[Float32],
        headKey
      )
      val timeDim = timeEncoderExtends.last
      val timeEncoder = MultiLayerPerceptron.Params.xavierUniform(
        Axis[Embedding],
        VType[Float32],
        timeEncoderExtends,
        timeEncoderKey
      )
      val bottleneckTimeParams = AffineLayer.Params.xavierUniform(
        Axis[Embedding] -> timeDim,
        Axis[Embedding] -> encoderExtents.last,
        VType[Float32],
        bottleneckTimeKey
      )
      val encoderTimeProjParams = encoderExtents.tail.zip(encoderTimeKey.split(encoderExtents.tail.size)).map:
        case (encoderOut, projLayerKey) =>
          AffineLayer.Params.xavierUniform(
            Axis[Embedding] -> timeDim,
            Axis[Embedding] -> encoderOut,
            VType[Float32],
            projLayerKey
          )
      val decodersTimeProjParams = decoderExtents.tail.zip(decoderTimeKey.split(decoderExtents.size)).map:
        case (decoderOut, projLayerKey) =>
          AffineLayer.Params.xavierUniform(
            Axis[Embedding] -> timeDim,
            Axis[Embedding] -> decoderOut,
            VType[Float32],
            projLayerKey
          )
      Params(
        stemParams,
        encoderParams,
        bottleneckSpatialParams,
        decoderParams,
        outputHeadParams,
        timeEncoder,
        bottleneckTimeParams,
        encoderTimeProjParams,
        decodersTimeProjParams
      )
