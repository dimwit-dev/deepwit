package deepwit.examples.mnistClassification

import dimwit.*

import deepwit.base.{AffineLayer, relu}
import deepwit.cnn.AffineConv2DLayer

object MNistCNN:

  case class Params(
      conv1: AffineConv2DLayer.Params[Height, Width, Channel, Hidden, Float32],
      conv2: AffineConv2DLayer.Params[Height, Width, Hidden, PixelEmbedding, Float32],
      output: AffineLayer.Params[ImageEmbedding, Output, Float32]
  )

  object Params:
    def apply(paramKey: Random.Key)(
        numHidden1: Int,
        numHidden2: Int
    ): Params =
      val (conv1Key, conv2Key, outputKey) = paramKey.splitToTuple(3)
      val kernelHeightDim = Axis[Height] -> 3
      val kernelWidthDim = Axis[Width] -> 3
      val channelDim = Axis[Channel] -> 1
      val hiddenDim = Axis[Hidden] -> numHidden1
      val pixelEmbeddingDim = Axis[PixelEmbedding] -> numHidden2
      val embeddingDim = Axis[ImageEmbedding] -> 7 * 7 * numHidden2
      val outputDim = Axis[Output] -> 10
      Params(
        conv1 = AffineConv2DLayer.Params.xavierUniform(kernelHeightDim, kernelWidthDim, channelDim, hiddenDim, VType[Float32], conv1Key),
        conv2 = AffineConv2DLayer.Params.xavierUniform(kernelHeightDim, kernelWidthDim, hiddenDim, pixelEmbeddingDim, VType[Float32], conv2Key),
        output = AffineLayer.Params.xavierUniform(embeddingDim, outputDim, VType[Float32], outputKey)
      )

class MNistCNN(params: MNistCNN.Params) extends (Tensor2[Height, Width, Float32] => Tensor0[Int32]):
  private val conv1 = AffineConv2DLayer(params.conv1, stride = 2, padding = Padding.SAME)
  private val conv2 = AffineConv2DLayer(params.conv2, stride = 2, padding = Padding.SAME)
  private val output = AffineLayer(params.output)

  def logits(image: Tensor2[Height, Width, Float32]): Tensor1[Output, Float32] =
    val input = image.appendAxis(Axis[Channel])
    val hidden = relu(conv1(input))
    val features = relu(conv2(hidden))
    output(features.flatten)

  override def apply(image: Tensor2[Height, Width, Float32]): Tensor0[Int32] = logits(image).argmax(Axis[Output])
