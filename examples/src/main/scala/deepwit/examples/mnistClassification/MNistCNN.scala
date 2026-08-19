package deepwit.examples.mnistClassification

import dimwit.*

import deepwit.base.AffineLayer
import deepwit.activation.relu
import deepwit.cnn.AffineConv2DLayer
import deepwit.regularization.Dropout

object MNistCNN:

  case class Params(
      conv1: AffineConv2DLayer.Params[Height, Width, Channel, Hidden, Float32],
      conv2: AffineConv2DLayer.Params[Height, Width, Hidden, PixelEmbedding, Float32],
      imageEmbeddingDropout: Dropout.Params[ImageEmbedding, Float32],
      output: AffineLayer.Params[ImageEmbedding, Output, Float32]
  ):

    /** The same parameters with the dropout projection thinned, as a training step wants them.
      *
      * Thinning at inference is what turns a single model into a Monte Carlo ensemble of itself.
      */
    def thinned(probability: Float, key: Key): Params =
      copy(imageEmbeddingDropout = imageEmbeddingDropout.thinned(probability, key))

  object Params:
    def apply(paramKey: Key)(
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
        conv1 = AffineConv2DLayer.Params.init(kernelHeightDim, kernelWidthDim, channelDim, hiddenDim, conv1Key),
        conv2 = AffineConv2DLayer.Params.init(kernelHeightDim, kernelWidthDim, hiddenDim, pixelEmbeddingDim, conv2Key),
        imageEmbeddingDropout = Dropout.Params.init(embeddingDim),
        output = AffineLayer.Params.init(embeddingDim, outputDim, outputKey)
      )

class MNistCNN(params: MNistCNN.Params) extends (Tensor2[Height, Width, Float32] => Tensor0[Int32]):
  private val conv1 = AffineConv2DLayer(params.conv1, stride = 2, padding = Padding.SAME)
  private val conv2 = AffineConv2DLayer(params.conv2, stride = 2, padding = Padding.SAME)
  private val imageEmbeddingDropout = Dropout(params.imageEmbeddingDropout)
  private val output = AffineLayer(params.output)

  def logits(image: Tensor2[Height, Width, Float32]): Tensor1[Output, Float32] =
    val input = image.appendAxis(Axis[Channel])
    val hidden = relu(conv1(input))
    val pixelEmbeddings = relu(conv2(hidden))
    val imageEmbedding = pixelEmbeddings.flatten
    output(imageEmbeddingDropout(imageEmbedding))

  override def apply(image: Tensor2[Height, Width, Float32]): Tensor0[Int32] = logits(image).argmax(Axis[Output])
