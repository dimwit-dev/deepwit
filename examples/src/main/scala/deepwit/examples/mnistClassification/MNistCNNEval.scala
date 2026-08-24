package deepwit.examples.mnistClassification

import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.examples.dataset.MNISTLoader

import dimwit.*
import dimwit.Conversions.given
import plotwit.*

import plotwit.PlotTargets.desktopBrowser

@main
def eval(): Unit =

  val runRoot = "out/MNistCNN"
  val checkpointer = TensorTreeCheckpointer.latestIn(runRoot).getOrElse(sys.error(s"Nothing to load: $runRoot holds no run yet. Train first."))
  val step = checkpointer.iterations.lastOption
  require(step.nonEmpty, s"No checkpoint to render in ${checkpointer.rootPath}.")
  println(s"Loading checkpoints from: ${checkpointer.rootPath}, reading the newest at step ${step.get}")

  val testDataset = MNISTLoader.createTestDataset().get
  val model = MNistCNN(checkpointer.loadLatest[TrainState].get.params)

  val imgGroups = testDataset.images.unstack(Axis[TestSample]).take(640).grouped(64).toList
  val labelGroups = testDataset.labels.unstack(Axis[TestSample]).take(640).grouped(64).toList

  val allPlots = imgGroups.zip(labelGroups).map: (imgs, labels) =>
    val imgsPlots = imgs.zip(labels).map: (img, label) =>
      val imgToPlot = (img.clip(0f, 1f) *! 255f).asInt(VType[UInt8]).swap(Axis[Height], Axis[Width])
      val prediction = model.logits(img).argmax(Axis[Output])
      plots.imagePlot(imgToPlot, _.title := f"T: $label, P: $prediction")
    grid(imgsPlots.grouped(8).toList)
  display(vconcat(allPlots))
