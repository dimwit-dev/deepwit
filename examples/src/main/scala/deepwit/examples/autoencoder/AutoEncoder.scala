package deepwit.examples.autoencoder

import dimwit.*

import deepwit.base.AffineLayer
import deepwit.activation.{relu, sigmoid}

class Autoencoder(params: Autoencoder.Params):

  val encoder = Autoencoder.Encoder(params.encoderParams)
  val decoder = Autoencoder.Decoder(params.decoderParams)

  def logits(x: Tensor1[Pixel, Float32]): Tensor1[ReconstructedPixel, Float32] = decoder.logits(encoder(x))
  def apply(x: Tensor1[Pixel, Float32]): Tensor1[ReconstructedPixel, Float32] = decoder(encoder(x))

object Autoencoder:

  case class Params(
      encoderParams: Encoder.Params,
      decoderParams: Decoder.Params
  )

  object Params:

    def xavierNormal(
        eHidden1Extent: AxisExtent[EHidden1],
        eHidden2Extent: AxisExtent[EHidden2],
        latentExtent: AxisExtent[Latent],
        dHidden1Extent: AxisExtent[DHidden1],
        dHidden2Extent: AxisExtent[DHidden2],
        key: Random.Key
    ): Params =
      val inputSize = 28 * 28
      val vtype = VType[Float32]
      val (encoderKey, decoderKey) = key.splitToTuple(2)
      val (encoderKey1, encoderKey2, encoderKey3) = encoderKey.splitToTuple(3)
      val encoderParams = Encoder.Params(
        AffineLayer.Params.xavierNormal(Axis[Pixel] -> inputSize, eHidden1Extent, vtype, encoderKey1),
        AffineLayer.Params.xavierNormal(eHidden1Extent, eHidden2Extent, vtype, encoderKey2),
        AffineLayer.Params.xavierNormal(eHidden2Extent, latentExtent, vtype, encoderKey3)
      )
      val (decoderKey1, decoderKey2, decoderKey3) = decoderKey.splitToTuple(3)
      val decoderParams = Decoder.Params(
        AffineLayer.Params.xavierNormal(latentExtent, dHidden1Extent, vtype, decoderKey1),
        AffineLayer.Params.xavierNormal(dHidden1Extent, dHidden2Extent, vtype, decoderKey2),
        AffineLayer.Params.xavierNormal(dHidden2Extent, Axis[ReconstructedPixel] -> inputSize, vtype, decoderKey3)
      )
      Params(encoderParams, decoderParams)

  class Encoder(p: Encoder.Params):

    val layer1 = AffineLayer(p.layer1)
    val layer2 = AffineLayer(p.layer2)
    val latentLayer = AffineLayer(p.latentLayer)

    def apply(v: Tensor1[Pixel, Float32]): Tensor1[Latent, Float32] =
      val h1 = relu(layer1(v))
      val h2 = relu(layer2(h1))
      latentLayer(h2)

  object Encoder:
    case class Params(
        layer1: AffineLayer.Params[Pixel, EHidden1, Float32],
        layer2: AffineLayer.Params[EHidden1, EHidden2, Float32],
        latentLayer: AffineLayer.Params[EHidden2, Latent, Float32]
    )

  class Decoder(p: Decoder.Params):

    val layer1 = AffineLayer(p.layer1)
    val layer2 = AffineLayer(p.layer2)
    val outputLayer = AffineLayer(p.outputLayer)

    def logits(v: Tensor1[Latent, Float32]): Tensor1[ReconstructedPixel, Float32] =
      val h1 = relu(layer1(v))
      val h2 = relu(layer2(h1))
      outputLayer(h2)

    def apply(v: Tensor1[Latent, Float32]): Tensor1[ReconstructedPixel, Float32] = sigmoid(logits(v))

  object Decoder:
    case class Params(
        layer1: AffineLayer.Params[Latent, DHidden1, Float32],
        layer2: AffineLayer.Params[DHidden1, DHidden2, Float32],
        outputLayer: AffineLayer.Params[DHidden2, ReconstructedPixel, Float32]
    )
