package deepwit.examples.autoencoder

import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import dimwit.*
import dimwit.Conversions.given
import plotwit.*

import plotwit.PlotTargets.desktopBrowser

@main
def eval(): Unit =

  val runRoot = "out/AutoEncoder"
  val checkpointer = TensorTreeCheckpointer.latestIn(runRoot).getOrElse(sys.error(s"Nothing to load: $runRoot holds no run yet. Train first."))
  val iterations = checkpointer.iterations
  require(iterations.nonEmpty, s"No checkpoint to render in ${checkpointer.rootPath}.")
  println(s"Loading checkpoints from: ${checkpointer.rootPath}, steps ${iterations.mkString(", ")}")

  val originals = MNISTLoader.createTestDataset().get.images
    .slice(Axis[TestSample].at(0 until 64))

  val originalsImgPanel = toImgPanel(originals)
  display(vconcat(
    iterations.map: step =>
      val model = Autoencoder(checkpointer.load[TrainState](step).get.params)
      val reconstruct = originals
        .vmap(Axis[TestSample])(sample => model.reconstruct(sample.flatten))
        .unflatten(Axis[Height |*| Width], Shape2(Axis[Height] -> 28, Axis[Width] -> 28))
      val reconstructionImgPanel = toImgPanel(reconstruct)
      hconcat(List(
        plots.imagePlot(originalsImgPanel, _.title := "Originals"),
        plots.imagePlot(reconstructionImgPanel, _.title := f"Reconstruction after $step steps")
      ))
  ))

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
