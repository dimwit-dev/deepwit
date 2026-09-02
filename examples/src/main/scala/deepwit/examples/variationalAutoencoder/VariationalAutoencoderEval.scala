package deepwit.examples.variationalAutoencoder

import dimwit.*
import dimwit.Conversions.given
import plotwit.*

import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import plotwit.PlotTargets.desktopBrowser

@main
def eval(): Unit =

  val runRoot = "out/VariationalAutoencoder"
  val checkpointer = TensorTreeCheckpointer.latestIn(runRoot).getOrElse(sys.error(s"Nothing to load: $runRoot holds no run yet. Train first."))
  val iterations = checkpointer.iterations
  require(iterations.nonEmpty, s"No checkpoint to render in ${checkpointer.rootPath}.")
  println(s"Loading checkpoints from: ${checkpointer.rootPath}, steps ${iterations.mkString(", ")}")

  val originals = MNISTLoader.createTestDataset().get.images
    .slice(Axis[TestSample].at(0 until 64))

  // Fixed once for the whole rendering, so every checkpoint is judged on the same draws and the
  // panels differ by what was learned rather than by what was sampled.
  val (reconstructionKey, generationKey) = Key(0).splitToTuple(2)
  val reconstructionKeys = reconstructionKey.splitToTensor(Axis[TestSample] -> 64)

  val originalsImgPanel = toImgPanel(originals)
  display(vconcat(
    iterations.map: step =>
      val model = VariationalAutoencoder(checkpointer.load[TrainState](step).get.params)
      val reconstructions = zipvmap(Axis[TestSample])(originals, reconstructionKeys): (sample, key) =>
        model.reconstruct(sample.flatten, key.item)
      val generations = generationKey.splitvmap(Axis[TestSample] -> 64)(model.generate)
      hconcat(List(
        plots.imagePlot(originalsImgPanel, _.title := "Originals"),
        plots.imagePlot(toImgPanel(toDigits(reconstructions)), _.title := f"Reconstruction after $step steps"),
        plots.imagePlot(toImgPanel(toDigits(generations)), _.title := f"Drawn from the prior after $step steps")
      ))
  ))

/** Reads a flat row of pixels back as the 28x28 image it stands for. */
private def toDigits(pixels: Tensor2[TestSample, ReconstructedPixel, Float32]): Tensor3[TestSample, Height, Width, Float32] =
  pixels.unflatten(Axis[Height |*| Width], Shape2(Axis[Height] -> 28, Axis[Width] -> 28))

/** Lays 64 samples out as an 8x8 tiling of the digits, so a whole batch reads as one image. */
private def toImgPanel(tensor: Tensor3[TestSample, Height, Width, Float32]): Tensor2[Prime[Width] |*| Width, Prime[Height] |*| Height, UInt8] =
  tensor
    .relabel(Axis[TestSample].as(Axis[Prime[Height] |*| Prime[Width]]))
    .rearrange(
      (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
      (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, Axis[Height] -> 28, Axis[Width] -> 28)
    )
    .clip(0f, 1f)
    .scale(255f)
    .asInt(VType[UInt8])
    .swap(Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width])
