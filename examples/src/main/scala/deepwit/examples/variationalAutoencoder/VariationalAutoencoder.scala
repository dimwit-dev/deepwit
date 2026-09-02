package deepwit.examples.variationalAutoencoder

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal

import deepwit.base.AffineLayer
import deepwit.activation.{relu, sigmoid}

class VariationalAutoencoder(params: VariationalAutoencoder.Params):

  val encoder = Encoder(params.encoderParams)
  val decoder = Decoder(params.decoderParams)

  val prior: Normal[Tuple1[Latent], Float32] = Normal.standardNormal(Shape1(encoder.latentExtent))

  def posterior(x: Tensor1[Pixel, Float32]): Normal[Tuple1[Latent], Float32] = encoder(x)

  def reconstruct(x: Tensor1[Pixel, Float32], key: Key): Tensor1[ReconstructedPixel, Float32] =
    sigmoid(decoder.logits(posterior(x).sample(key)))

  def generate(key: Key): Tensor1[ReconstructedPixel, Float32] =
    sigmoid(decoder.logits(prior.sample(key)))

object VariationalAutoencoder:

  case class Params(
      encoderParams: Encoder.Params,
      decoderParams: Decoder.Params
  )

  object Params:

    def init(
        eHidden1Extent: AxisExtent[EHidden1],
        eHidden2Extent: AxisExtent[EHidden2],
        latentExtent: AxisExtent[Latent],
        dHidden1Extent: AxisExtent[DHidden1],
        dHidden2Extent: AxisExtent[DHidden2],
        key: Key
    ): Params =
      val inputSize = 28 * 28
      val (encoderKey, decoderKey) = key.splitToTuple(2)
      val (encoderKey1, encoderKey2, meanKey, logVarianceKey) = encoderKey.splitToTuple(4)
      val encoderParams = Encoder.Params(
        AffineLayer.Params.init(Axis[Pixel] -> inputSize, eHidden1Extent, encoderKey1),
        AffineLayer.Params.init(eHidden1Extent, eHidden2Extent, encoderKey2),
        AffineLayer.Params.init(eHidden2Extent, latentExtent, meanKey),
        AffineLayer.Params.init(eHidden2Extent, latentExtent, logVarianceKey)
      )
      val (decoderKey1, decoderKey2, decoderKey3) = decoderKey.splitToTuple(3)
      val decoderParams = Decoder.Params(
        AffineLayer.Params.init(latentExtent, dHidden1Extent, decoderKey1),
        AffineLayer.Params.init(dHidden1Extent, dHidden2Extent, decoderKey2),
        AffineLayer.Params.init(dHidden2Extent, Axis[ReconstructedPixel] -> inputSize, decoderKey3)
      )
      Params(encoderParams, decoderParams)

class Encoder(p: Encoder.Params):

  val layer1 = AffineLayer(p.layer1)
  val layer2 = AffineLayer(p.layer2)
  val meanLayer = AffineLayer(p.meanLayer)
  val logVarianceLayer = AffineLayer(p.logVarianceLayer)
  val latentExtent: AxisExtent[Latent] = Axis[Latent] -> p.meanLayer.bias.shape(Axis[Latent])

  def apply(v: Tensor1[Pixel, Float32]): Normal[Tuple1[Latent], Float32] =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val std = (logVarianceLayer(h2).clip(-10f, 10f) *! 0.5f).exp
    Normal(meanLayer(h2), std)

object Encoder:

  case class Params(
      layer1: AffineLayer.Params[Pixel, EHidden1, Float32],
      layer2: AffineLayer.Params[EHidden1, EHidden2, Float32],
      meanLayer: AffineLayer.Params[EHidden2, Latent, Float32],
      logVarianceLayer: AffineLayer.Params[EHidden2, Latent, Float32]
  )

class Decoder(params: Decoder.Params):

  val layer1 = AffineLayer(params.layer1)
  val layer2 = AffineLayer(params.layer2)
  val outputLayer = AffineLayer(params.outputLayer)

  def logits(v: Tensor1[Latent, Float32]): Tensor1[ReconstructedPixel, Float32] =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    outputLayer(h2)

object Decoder:

  case class Params(
      layer1: AffineLayer.Params[Latent, DHidden1, Float32],
      layer2: AffineLayer.Params[DHidden1, DHidden2, Float32],
      outputLayer: AffineLayer.Params[DHidden2, ReconstructedPixel, Float32]
  )
