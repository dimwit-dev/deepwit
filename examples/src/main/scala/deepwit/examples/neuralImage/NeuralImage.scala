package deepwit.examples.neuralImage

import dimwit.*

import deepwit.base.AffineLayer
import deepwit.activation.{relu, sigmoid}

/** An image held as a function from one pixel coordinate to its colour, rather than as a grid. */
class NeuralImage(params: NeuralImage.Params) extends (Tensor1[PixelCoordinate, Float32] => Tensor1[Channel, Float32]):

  private val layer1 = AffineLayer(params.layer1)
  private val layer2 = AffineLayer(params.layer2)
  private val layer3 = AffineLayer(params.layer3)
  private val output = AffineLayer(params.output)

  override def apply(coordinate: Tensor1[PixelCoordinate, Float32]): Tensor1[Channel, Float32] =
    val h1 = relu(layer1(coordinate))
    val h2 = relu(layer2(h1))
    val h3 = relu(layer3(h2))
    sigmoid(output(h3))

object NeuralImage:

  case class Params(
      layer1: AffineLayer.Params[PixelCoordinate, Hidden1, Float32],
      layer2: AffineLayer.Params[Hidden1, Hidden2, Float32],
      layer3: AffineLayer.Params[Hidden2, Hidden3, Float32],
      output: AffineLayer.Params[Hidden3, Channel, Float32]
  )

  object Params:

    def init(
        coordinateExtent: AxisExtent[PixelCoordinate],
        hiddenSize: Int,
        channelExtent: AxisExtent[Channel],
        key: Key
    ): Params =
      val (key1, key2, key3, key4) = key.splitToTuple(4)
      Params(
        layer1 = AffineLayer.Params.init(coordinateExtent, Axis[Hidden1] -> hiddenSize, key1),
        layer2 = AffineLayer.Params.init(Axis[Hidden1] -> hiddenSize, Axis[Hidden2] -> hiddenSize, key2),
        layer3 = AffineLayer.Params.init(Axis[Hidden2] -> hiddenSize, Axis[Hidden3] -> hiddenSize, key3),
        output = AffineLayer.Params.init(Axis[Hidden3] -> hiddenSize, channelExtent, key4)
      )
