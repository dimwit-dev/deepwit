package deepwit.examples.autoencoder

import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import dimwit.*
import dimwit.Conversions.given
import dimwit.python.PyBridge.toPyTensor

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

def toImg2D(tensor: Tensor[(TestSample, Height, Width), Float32]): Tensor[(Prime[Height] |*| Height, Prime[Width] |*| Width), Float32] =
  tensor
    .relabel(Axis[TestSample].as(Axis[Prime[Height] |*| Prime[Width]]))
    .rearrange(
      (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
      (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, Axis[Height] -> 28, Axis[Width] -> 28)
    )

@main
/** Opens an interactive Matplotlib dashboard to evaluate an Autoencoder's performance.
  *
  * The UI displays three side-by-side image grids:
  * 1. '''Original''': The baseline MNIST test samples.
  * 2. '''Reconstruction''': The model's output at a specific training checkpoint.
  * 3. '''Latent Traversal''': The reconstruction after manually tweaking a specific
  * latent dimension by a user-defined epsilon.
  *
  * Interactive controls (sliders) allow the user to scrub through training
  * iterations, select latent dimensions, and adjust the traversal intensity
  * in real-time.
  *
  * @param checkpointPath The path to the checkpoint containing the AutoEncoder TrainState.
  */
def autoEncoderEval(checkpointPath: String) =

  println(s"Loading checkpoints from: $checkpointPath")

  val matplotlib = py.module("matplotlib")
  matplotlib.use("WebAgg")
  val plt = py.module("matplotlib.pyplot")
  val widgets = py.module("matplotlib.widgets")

  // ... (Keep your toImg2D and MNISTLoader logic here) ...
  val testDataset = MNISTLoader.createTestDataset().get
  val original = testDataset.images.slice(Axis[TestSample].at(0 until 64))
  val originalImg = toPyTensor(toImg2D(original))

  val emptyPlaceholderImg = toPyTensor(toImg2D(Tensor.like(original).fill(0f)))

  val logger = new TensorTreeCheckpointer(checkpointPath)
  val iterations = logger.iterations

  // 2. Setup Figure and Subplots
  val fig = plt.figure(figsize = List(18, 7).toPythonCopy)
  plt.subplots_adjust(bottom = 0.35) // Extra space for 3 sliders

  val axOrig = fig.add_subplot(1, 3, 1)
  axOrig.set_title("Original")
  axOrig.imshow(originalImg, cmap = "gray")

  val axRec = fig.add_subplot(1, 3, 2)
  val recDisplay = axRec.imshow(emptyPlaceholderImg, cmap = "gray", vmin = 0, vmax = 1)
  axRec.set_title(s"Iteration ${iterations.head}")

  val axTraversal = fig.add_subplot(1, 3, 3)
  val traversalDisplay = axTraversal.imshow(emptyPlaceholderImg, cmap = "gray", vmin = 0, vmax = 1)
  axTraversal.set_title("Latent Traversal")

  val axIter = fig.add_axes(List(0.25, 0.2, 0.5, 0.03).toPythonCopy)
  val sIter = widgets.Slider(axIter, "Iteration", iterations.min.toDouble, iterations.max.toDouble, valstep = iterations.toPythonCopy)

  val axDim = fig.add_axes(List(0.25, 0.15, 0.5, 0.03).toPythonCopy)
  val sDim = widgets.Slider(axDim, "Latent Dim", 0, 19, valinit = 0, valstep = 1)

  val axEps = fig.add_axes(List(0.25, 0.1, 0.5, 0.03).toPythonCopy)
  val sEps = widgets.Slider(axEps, "Epsilon", 0.0, 100.0, valinit = 0.0)

  // 4. Define Update Logic
  var model = Autoencoder(logger.load[TrainState](iterations.head).get.params)
  var currentIt = iterations.head
  val update = (valVal: py.Any) =>
    val it = sIter.`val`.as[Double].toInt
    val dimIdx = sDim.`val`.as[Double].toInt
    val eps = sEps.`val`.as[Float]

    val state = logger.load[TrainState](it).get
    if currentIt != it then
      // Only load model if iteration has changed, avoid redundant loading
      println(s"Loading checkpoint for iteration: $it")
      currentIt = it
      model = Autoencoder(state.params)

    val rec = original.vmap(Axis[TestSample]): sample =>
      model(sample.flatten)
    .unflatten(Axis[Height |*| Width], Shape2(Axis[Height] -> 28, Axis[Width] -> 28))

    // Latent Traversal
    val traversal = original.vmap(Axis[TestSample]): sample =>
      val latent = model.encoder(sample.flatten)
      val value = latent.slice(Axis[Latent].at(dimIdx))
      val latentTraversed = latent.set(Axis[Latent].at(dimIdx))(value + eps)
      model.decoder(latentTraversed)
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
