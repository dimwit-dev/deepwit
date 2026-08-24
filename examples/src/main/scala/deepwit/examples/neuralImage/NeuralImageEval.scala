package deepwit.examples.neuralImage

import dimwit.*
import dimwit.Conversions.given
import plotwit.*

import deepwit.checkpointing.TensorTreeCheckpointer

import plotwit.PlotTargets.desktopBrowser

@main
def eval(): Unit =

  val imageSize = 256

  val runRoot = "out/NeuralImage"
  val checkpointer = TensorTreeCheckpointer.latestIn(runRoot).getOrElse(sys.error(s"Nothing to load: $runRoot holds no run yet. Train first."))
  val iteration = checkpointer.iterations.lastOption
  require(iteration.nonEmpty, s"No checkpoint to render in ${checkpointer.rootPath}.")

  val state = checkpointer.loadLatest[TrainState].get
  // The network's first layer says how many numbers it reads a position as, so the encoding handed
  // to it now is rebuilt to the width it was fitted with rather than to a constant repeated here.
  val encodingSize = state.params.layer1.weight.shape(Axis[PixelCoordinate])
  println(f"Loaded ${checkpointer.rootPath} after ${iteration.get} steps, reading a position as $encodingSize numbers, at cost ${state.lastCost.item}%.6f.")

  val image = ImageLoader.load(imageSize)
  val heightExtent = Axis[Height] -> imageSize
  val widthExtent = Axis[Width] -> imageSize
  val grid = Shape(heightExtent, widthExtent)
  val coordinates = Coordinates.fourier(Coordinates.positions(heightExtent), Coordinates.positions(widthExtent), encodingSize)
  val fourierImage = render(state.params, coordinates, grid)

  display(
    hconcat(List(
      plots.imagePlot(prep(image), _.title := "target"),
      plots.imagePlot(prep(fourierImage), _.title := f"$encodingSize Fourier features\ncost ${state.lastCost.item}%.5f")
    ))
  )

  // A window on the dog's face, held as fractions of the image so it survives a change of imageSize.
  val cropTop = (0.03f * imageSize).round
  val cropLeft = (0.35f * imageSize).round
  val cropSize = (0.25f * imageSize).round
  val zoom = 4

  val trainedCrop = fourierImage
    .slice(Axis[Height].at((cropTop until cropTop + cropSize).toIndexedSeq))
    .slice(Axis[Width].at((cropLeft until cropLeft + cropSize).toIndexedSeq))

  // The network never saw a position between two pixels; the encoding is defined at them all the same.
  val zoomedSize = cropSize * zoom
  val zoomedCoordinates = Coordinates.fourier(
    Coordinates.positions[Height](cropTop.toFloat, cropTop + cropSize - 1f, zoomedSize),
    Coordinates.positions[Width](cropLeft.toFloat, cropLeft + cropSize - 1f, zoomedSize),
    encodingSize
  )
  val zoomedGrid = Shape(Axis[Height] -> zoomedSize, Axis[Width] -> zoomedSize)

  display(
    hconcat(List(
      plots.imagePlot(prep(nearest(trainedCrop, zoom)), _.title := f"trained crop ${cropSize}x$cropSize, pixels repeated"),
      plots.imagePlot(prep(render(state.params, zoomedCoordinates, zoomedGrid)), _.title := f"network zoom ${zoomedSize}x$zoomedSize")
    ))
  )

private def prep(img: Tensor3[Height, Width, Channel, Float32]): Tensor3[Width, Height, Channel, UInt8] =
  (img.clip(0f, 1f) *! 255f).asInt(VType[UInt8])
    .swap(Axis[Height], Axis[Width])

/** Evaluates the network at every position it is given, turning the function back into an image. */
private def render(
    params: NeuralImage.Params,
    coordinates: Tensor2[Pixel, PixelCoordinate, Float32],
    grid: Shape2[Height, Width]
): Tensor3[Height, Width, Channel, Float32] =
  coordinates.vmap(Axis[Pixel])(NeuralImage(params)).unflatten(Axis[Pixel], grid)

/** Repeats each pixel `zoom` times along both axes, which adds nothing to an image but the size it
  * is drawn at: the way to show a small one beside a large one without claiming to have zoomed it.
  */
private def nearest(image: Tensor3[Height, Width, Channel, Float32], zoom: Int): Tensor3[Height, Width, Channel, Float32] =
  def repeated(extent: Int) = (0 until extent * zoom).map(_ / zoom).toIndexedSeq
  image
    .slice(Axis[Height].at(repeated(image.shape(Axis[Height]))))
    .slice(Axis[Width].at(repeated(image.shape(Axis[Width]))))
