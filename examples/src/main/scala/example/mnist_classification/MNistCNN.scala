package deepwit.example.mnist_classification

import dimwit.*
import dimwit.Conversions.given
import deepwit.*
import nn.ActivationFunctions.relu

object MNistCNN:

  case class Params(
      conv1: AffineConv2DLayer.Params[Height, Width, Channel, Hidden],
      conv2: AffineConv2DLayer.Params[Height, Width, Hidden, PixelEmbedding],
      output: AffineLayer.Params[ImageEmbedding, Output]
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
        conv1 = AffineConv2DLayer.Params.xavierUniform(kernelHeightDim, kernelWidthDim, channelDim, hiddenDim, conv1Key),
        conv2 = AffineConv2DLayer.Params.xavierUniform(kernelHeightDim, kernelWidthDim, hiddenDim, pixelEmbeddingDim, conv2Key),
        output = AffineLayer.Params.xavierUniform(embeddingDim, outputDim, outputKey)
      )

case class MNistCNN(params: MNistCNN.Params) extends Function[Tensor2[Height, Width, Float], Tensor0[Int]]:
  private val conv1 = AffineConv2DLayer(Conv2DLayer.HyperParams(stride = 2, padding = Padding.SAME))(params.conv1)
  private val conv2 = AffineConv2DLayer(Conv2DLayer.HyperParams(stride = 2, padding = Padding.SAME))(params.conv2)
  private val output = AffineLayer(params.output)

  def logits(image: Tensor2[Height, Width, Float]): Tensor1[Output, Float] =
    val input = image.appendAxis(Axis[Channel])
    val hidden = relu(conv1(input))
    val features = relu(conv2(hidden))
    output(features.flatten)

  override def apply(image: Tensor2[Height, Width, Float]): Tensor0[Int] =
    logits(image).argmax(Axis[Output])
