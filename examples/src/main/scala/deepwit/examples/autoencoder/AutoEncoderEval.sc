import dimwit.*
import dimwit.Conversions.given
import deepwit.logging.TensorTreeLogger
import deepwit.examples.autoencoder.*

import deepwit.examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import plotwit.plotting.Plotting
import plotwit.plotting.Plotting.{hconcat, grid}

// import viz.PlotTargets.websocket
import viz.PlotTargets.desktopBrowser

dimwit.initialize()

val logger = new TensorTreeLogger("/Users/mebr/Documents/Privat/Projects/deepwit/examples/out/AutoEncoder/20260802_163544")
val state = logger.load[TrainState](2000).get
val model = Autoencoder(state.params)

val testDataset = MNISTLoader.createTestDataset().get
val original = testDataset.images.slice(Axis[TestSample].at(0 until 64))

val rec = original.vmap(Axis[TestSample]): sample =>
  model(sample.flatten)
    .unflatten(Axis[Height |*| Width], Shape2(Axis[Height] -> 28, Axis[Width] -> 28))

println(original.shape.toString)
println(rec.shape.toString)

val specs = original.unstack(Axis[TestSample]).zip(rec.unstack(Axis[TestSample])).map:
  case (original, rec) =>
    def toUInt8Img(img: Tensor2[Height, Width, Float32]): Tensor2[Height, Width, UInt8] =
      ((img -! img.min) /! (img.max - img.min) *! 255.0f).asInt(VType[UInt8])
    hconcat(Seq(
      Plotting.image.plot(toUInt8Img(original).transpose, _.mark.width := 56, _.mark.height := 56, _.title := ""),
      Plotting.image.plot(toUInt8Img(rec).transpose, _.mark.width := 56, _.mark.height := 56, _.title := "")
    ))

Plotting.display(grid(specs.grouped(8).toSeq))
