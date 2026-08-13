package deepwit.example.mnist_classification

import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.examples.dataset.MNISTLoader
import MNISTLoader.{TestSample, Height, Width}

import dimwit.*
import dimwit.Conversions.given
import dimwit.python.PyBridge.toPyTensor
import deepwit.loss.CategoricalCrossEntropy
import deepwit.base.softmax

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

@main
def mnistCNNEval(checkpointPath: String) =
  val np = py.module("numpy")
  val plt = py.module("matplotlib.pyplot")
  val widgets = py.module("matplotlib.widgets")

  // 1. Setup Data
  val testDataset = MNISTLoader.createTestDataset().get
  val checkpointer = new TensorTreeCheckpointer(checkpointPath)
  val iterations = checkpointer.iterations

  if iterations.isEmpty then throw new RuntimeException("No checkpoints found!")

  // 2. State Management
  var currentModel: MNistCNN = null
  var allLogits: Tensor[(TestSample, Output), Float32] = null
  var allPredictions: Tensor1[TestSample, Int32] = null
  var sortedIndices: Seq[Int] = Nil
  var randomIndices: Seq[Int] = scala.util.Random.shuffle(0 until testDataset.images.shape(Axis[TestSample]))
  var accuracy: Float = 0f

  // 3. Visualization Setup
  val fig = plt.figure(figsize = List(18, 10).toPythonCopy)
  plt.subplots_adjust(bottom = 0.25, hspace = 0.4)

  val axWorst = fig.add_subplot(2, 2, 1)
  val axRandom = fig.add_subplot(2, 2, 2)
  val axDetail = fig.add_subplot(2, 2, 3)
  val axProbs = fig.add_subplot(2, 2, 4)

  // 4. Update Logic
  def updateIteration(it: Int): Unit =
    val state = checkpointer.load[TrainState](it).get
    currentModel = MNistCNN(state.params)

    // Calculate global metrics for this iteration
    allLogits = testDataset.images.vmap(Axis[TestSample])(currentModel.logits)
    allPredictions = allLogits.vmap(Axis[TestSample])(_.argmax(Axis[Output]))

    // Calculate Accuracy
    val matches = zipvmap(Axis[TestSample])(allPredictions, testDataset.labels)(_ === _)
    accuracy = matches.asFloat32.mean.item * 100f

    // Sort by Loss for "Worst" mistakes
    val losses = zipvmap(Axis[TestSample])(testDataset.labels, allLogits): (target, logits) =>
      CategoricalCrossEntropy.fromLogits(target, logits)

    val pyLosses = toPyTensor(losses)
    sortedIndices = np.argsort(-pyLosses).as[Seq[Int]]

    refreshUI(it)

  def refreshUI(it: Int): Unit =
    fig.suptitle(f"MNIST CNN Evaluation | Iteration: $it | Overall Accuracy: $accuracy%.2f%%", fontsize = 16)

    plotGridWithLabels(axWorst, sortedIndices, "Worst Mistakes (Highest Loss)")
    plotGridWithLabels(axRandom, randomIndices, "Random Samples")

    // Draw the top "worst" by default
    drawDetail(sortedIndices.head)
    fig.canvas.draw_idle()

  def drawDetail(idx: Int): Unit =
    val img = testDataset.images.slice(Axis[TestSample].at(idx))
    val target = testDataset.labels.slice(Axis[TestSample].at(idx)).item
    val pred = allPredictions.slice(Axis[TestSample].at(idx)).item
    val logits = allLogits.slice(Axis[TestSample].at(idx))
    val probs = softmax(logits)

    axDetail.clear()
    axDetail.imshow(toPyTensor(img), cmap = "gray")
    axDetail.set_title(s"Target: $target | Pred: $pred")

    axProbs.clear()
    axProbs.bar((0 to 9).toPythonCopy, toPyTensor(probs))
    axProbs.set_xticks((0 to 9).toPythonCopy)
    axProbs.set_ylim(0, 1)
    axProbs.set_title("Softmax Confidence")

  def plotGridWithLabels(ax: py.Dynamic, indices: Seq[Int], title: String): Unit =
    ax.clear()
    val gridIndices = indices.take(16)
    val gridImages = gridIndices.map(idx => testDataset.images.slice(Axis[TestSample].at(idx)))
    val combined = stack(gridImages, Axis[TestSample])

    ax.imshow(toPyTensor(toImg2D_Small(combined)), cmap = "gray")
    ax.set_title(title)
    ax.set_axis_off()

    val bboxParams = py.Dynamic.global.dict(Seq(
      "facecolor" -> ("white": py.Any),
      "alpha" -> (0.7: py.Any),
      "pad" -> (0: py.Any)
    ).toPythonCopy)

    for i <- 0 until 16 do
      val idx = gridIndices(i)
      val target = testDataset.labels.slice(Axis[TestSample].at(idx)).item
      val pred = allPredictions.slice(Axis[TestSample].at(idx)).item
      val color = if target == pred then "green" else "red"

      ax.text(
        (i % 4) * 28 + 1,
        (i / 4) * 28 + 2,
        s"T:$target P:$pred",
        color = color,
        fontsize = 7,
        fontweight = "bold",
        bbox = bboxParams,
        verticalalignment = "top"
      )

  // 5. Sliders
  val axIter = fig.add_axes(List(0.25, 0.15, 0.5, 0.03).toPythonCopy)
  val sIter = widgets.Slider(
    axIter,
    "Step",
    iterations.min.toDouble,
    iterations.max.toDouble,
    valinit = iterations.max.toDouble,
    valstep = iterations.toPythonCopy
  )

  val axWorstIdx = fig.add_axes(List(0.25, 0.1, 0.5, 0.03).toPythonCopy)
  val sWorst = widgets.Slider(axWorstIdx, "Mistake #", 0, 100, valinit = 0, valstep = 1)

  // 6. Events
  sIter.on_changed((v: py.Any) => updateIteration(v.as[Double].toInt))
  sWorst.on_changed((v: py.Any) =>
    drawDetail(sortedIndices(v.as[Double].toInt))
    fig.canvas.draw_idle()
  )

  // Start with the latest checkpoint
  updateIteration(iterations.max)
  plt.show()

def toImg2D_Small(tensor: Tensor[(TestSample, Height, Width), Float32]): Tensor[(Prime[Height] |*| Height, Prime[Width] |*| Width), Float32] =
  tensor.relabel(Axis[TestSample] -> Axis[Prime[Height] |*| Prime[Width]])
    .rearrange(
      (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
      (Axis[Prime[Height]] -> 4, Axis[Prime[Width]] -> 4, Axis[Height] -> 28, Axis[Width] -> 28)
    )
