package deepwit.examples.mnistClassification

import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.examples.newestRun
import deepwit.examples.dataset.MNISTLoader

import dimwit.*
import dimwit.Conversions.given
import deepwit.loss.CategoricalCrossEntropy
import deepwit.activation.softmax
import plotwit.*

import plotwit.PlotTargets.desktopBrowser

@main
def eval(givenPath: String*): Unit =

  val checkpointPath = givenPath.headOption.getOrElse(newestRun("out/MNistCNN"))
  val checkpointer = TensorTreeCheckpointer(checkpointPath)
  val iterations = checkpointer.iterations
  require(iterations.nonEmpty, s"No checkpoint to render in $checkpointPath.")

  val step = iterations.max
  println(s"Loading checkpoints from: $checkpointPath, reading the newest at step $step")

  val testDataset = MNISTLoader.createTestDataset().get
  val model = MNistCNN(checkpointer.load[TrainState](step).get.params)

  val imgGroups = testDataset.images.unstack(Axis[TestSample]).take(640).grouped(64).toList
  val labelGroups = testDataset.labels.unstack(Axis[TestSample]).take(640).grouped(64).toList

  val allPlots = imgGroups.zip(labelGroups).map: (imgs, labels) =>
    val imgsPlots = imgs.zip(labels).map: (img, label) =>
      val imgToPlot = (img.clip(0f, 1f) *! 255f).asInt(VType[UInt8]).swap(Axis[Height], Axis[Width])
      val prediction = model.logits(img).argmax(Axis[Output])
      plots.imagePlot(imgToPlot, _.title := f"T: $label, P: $prediction")
    grid(imgsPlots.grouped(8).toList)
  display(vconcat(allPlots))
