package deepwit.examples.neuralImage

import dimwit.*
import dimwit.{Label, |*|}
import deepwit.embedder.PositionalEncoding

/** The way of handing a pixel position to a network that has no notion of space.
  *
  * Positions are the grid's own indices, so a model trained on one grid can be asked about any
  * other: a finer sampling of it, or a window cut out of it.
  */
object Coordinates:

  def positions[P: Label](extent: AxisExtent[P]): Tensor1[P, Float32] =
    positions[P](0, extent.size - 1, extent.size)

  def positions[P: Label](from: Float, to: Float, count: Int): Tensor1[P, Float32] =
    require(count > 0, s"A grid needs at least one position, but was asked for $count.")
    val step = if count == 1 then 0f else (to - from) / (count - 1)
    Tensor(Shape1(Axis[P] -> count), VType[Float32]).fromArray(Array.tabulate(count)(i => from + i * step))

  /** Each position of the plane as a bank of sines and cosines of it, at geometrically spaced
    * frequencies, laid out along the axis the image is a function over.
    */
  def fourier(
      heightPositions: Tensor1[Height, Float32],
      widthPositions: Tensor1[Width, Float32],
      encodingSize: Int
  ): Tensor2[Pixel, PixelCoordinate, Float32] =
    PositionalEncoding
      .sinusoidal2D(heightPositions, widthPositions, Axis[PixelCoordinate] -> encodingSize)
      .flatten((Axis[Height], Axis[Width]))
