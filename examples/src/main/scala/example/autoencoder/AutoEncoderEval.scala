package example.autoencoder

import deepwit.logging.TenZarrLogger
import examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import dimwit.*
import dimwit.Conversions.given
import dimwit.python.PyBridge.toPyTensor

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

def toImg2D(tensor: Tensor[(TestSample, Height, Width), Float]): Tensor[(Prime[Height] |*| Height, Prime[Width] |*| Width), Float] =
  tensor
    .relabel(Axis[TestSample].as(Axis[Prime[Height] |*| Prime[Width]]))
    .rearrange(
      (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
      (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, Axis[Height] -> 28, Axis[Width] -> 28)
    )

@main
def autoEncoderEval(checkpointFolder: String) =
  val plt = py.module("matplotlib.pyplot")
  val widgets = py.module("matplotlib.widgets")

  // ... (Keep your toImg2D and MNISTLoader logic here) ...
  val (testX, testY) = MNISTLoader.createTestDataset().get
  val original = testX.slice(Axis[TestSample].at(0 until 64))
  val originalImg = toPyTensor(toImg2D(original))

  val logger = new TenZarrLogger(f"out/AutoEncoder/$checkpointFolder")
  val iterations = logger.iterations("checkpoint")

  // 2. Setup Figure and Subplots
  val fig = plt.figure(figsize = List(18, 7).toPythonCopy)
  plt.subplots_adjust(bottom = 0.35) // Extra space for 3 sliders

  val axOrig = fig.add_subplot(1, 3, 1)
  axOrig.set_title("Original")
  axOrig.imshow(originalImg, cmap = "gray")

  val axRec = fig.add_subplot(1, 3, 2)
  val recDisplay = axRec.imshow(originalImg, cmap = "gray")
  axRec.set_title(s"Iteration ${iterations.head}")

  val axTraversal = fig.add_subplot(1, 3, 3)
  val traversalDisplay = axTraversal.imshow(originalImg, cmap = "gray") // Placeholder
  axTraversal.set_title("Latent Traversal")

  val axIter = fig.add_axes(List(0.25, 0.2, 0.5, 0.03).toPythonCopy)
  val sIter = widgets.Slider(axIter, "Iteration", iterations.min.toDouble, iterations.max.toDouble, valstep = 500.0)

  val axDim = fig.add_axes(List(0.25, 0.15, 0.5, 0.03).toPythonCopy)
  val sDim = widgets.Slider(axDim, "Latent Dim", 0, 19, valinit = 0, valstep = 1)

  val axEps = fig.add_axes(List(0.25, 0.1, 0.5, 0.03).toPythonCopy)
  val sEps = widgets.Slider(axEps, "Epsilon", 0.0, 100.0, valinit = 0.0)

  // 4. Define Update Logic
  val update = (valVal: py.Any) =>
    val it = sIter.`val`.as[Double].toInt
    val dimIdx = sDim.`val`.as[Double].toInt
    val eps = sEps.`val`.as[Float]

    val state = logger.loadTensorTree[TrainState]("checkpoint", iteration = it).get
    val model = Autoencoder(state.params)

    val rec = original.vmap(Axis[TestSample]): sample =>
      model(sample.flatten)
    .unflatten(Axis[Height |*| Width], Shape2(Axis[Height] -> 28, Axis[Width] -> 28))

    // Latent Traversal
    val traversal = original.vmap(Axis[TestSample]): sample =>
      val latent = model.encoder(sample.flatten)
      val value = latent.slice(Axis[Latent].at(dimIdx))
      val latentTraversed = latent.set(Axis[Latent].at(dimIdx))(value + eps)
      model.decode(latentTraversed)
    .unflatten(Axis[Height |*| Width], Shape2(Axis[Height] -> 28, Axis[Width] -> 28))

    recDisplay.set_data(toPyTensor(toImg2D(rec)))
    traversalDisplay.set_data(toPyTensor(toImg2D(traversal)))
    axRec.set_title(s"Iteration $it")
    fig.canvas.draw_idle()

  // Connect the slider to the update function
  sIter.on_changed(update)
  sDim.on_changed(update)
  sEps.on_changed(update)

  plt.show()

import me.shadaj.scalapy.py
import example.autoencoder as i
import example.autoencoder as logger
