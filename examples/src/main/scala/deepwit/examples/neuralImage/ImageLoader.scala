package deepwit.examples.neuralImage

import dimwit.*

import java.awt.image.BufferedImage
import javax.imageio.ImageIO

/** Loads the photograph the example descends on as a square grid of channel intensities in [0, 1].
  *
  * A corgi on cobblestones in front of an out-of-focus hedge, 512 x 512: fur and stone carry the
  * high frequencies, the blurred background the low ones, so both are present and easy to tell
  * apart.
  *
  * Wikimedia Commons, "Fawn and white Welsh Corgi puppy standing on rear legs and sticking out the
  * tongue", released under CC0, cropped to a square and resampled.
  */
object ImageLoader:

  /** Reads the photograph onto a `size` x `size` grid. */
  def load(size: Int): Tensor3[Height, Width, Channel, Float32] =
    require(size > 0, s"An image needs at least one pixel per side, but was asked for $size.")
    val stream = getClass.getResourceAsStream("/images/dog.jpg")
    val image =
      try ImageIO.read(stream)
      finally stream.close()
    squareIntensities(image, size)

  /** Cuts the largest centred square out of the image and samples it onto a `size` x `size` grid.
    *
    * Each output pixel averages the source pixels it covers rather than picking one of them, so that
    * shrinking a photograph does not alias away the very detail an image example is about. Asked for
    * more pixels than the source has, the same rule reads the nearest one.
    */
  private def squareIntensities(image: BufferedImage, size: Int): Tensor3[Height, Width, Channel, Float32] =
    val side = math.min(image.getWidth, image.getHeight)
    val left = (image.getWidth - side) / 2
    val top = (image.getHeight - side) / 2
    val pixels = image.getRGB(left, top, side, side, null, 0, side)

    val values = new Array[Float](size * size * 3)
    for h <- 0 until size; w <- 0 until size do
      val fromRow = h * side / size
      val toRow = math.max(fromRow + 1, (h + 1) * side / size)
      val fromCol = w * side / size
      val toCol = math.max(fromCol + 1, (w + 1) * side / size)
      var red, green, blue = 0
      for row <- fromRow until toRow; col <- fromCol until toCol do
        val rgb = pixels(row * side + col)
        red += (rgb >> 16) & 0xff
        green += (rgb >> 8) & 0xff
        blue += rgb & 0xff
      val covered = (toRow - fromRow) * (toCol - fromCol) * 255f
      val at = (h * size + w) * 3
      values(at) = red / covered
      values(at + 1) = green / covered
      values(at + 2) = blue / covered
    Tensor(Shape(Axis[Height] -> size, Axis[Width] -> size, Axis[Channel] -> 3), VType[Float32]).fromArray(values)
