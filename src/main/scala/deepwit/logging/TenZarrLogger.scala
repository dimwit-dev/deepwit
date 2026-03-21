package deepwit.logging

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.*
import dimwit.python.PyBridge.{toPyTensor, liftPyTensor, liftPyTensor1}
import deepwit.example.MLPParams

trait Iteration derives Label

class TenZarrLogger(storePath: String = "logs.zarr"):

  private val zarr = py.module("zarr")
  private val np = py.module("numpy")
  private val root = zarr.open(storePath, mode = "a")

  def logTree[Data: TensorTree](name: String, iteration: Int, data: Data): Unit =
    TensorTree.foreach(
      data,
      [T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (tensorName: String, tensor: Tensor[T, V]) =>
            log(s"$name/$tensorName", iteration, tensor)
    )

  def log[T <: Tuple, V](name: String, iteration: Int, data: Tensor[T, V]): Unit =
    val pyData = np.array(toPyTensor(data))
    val dimSizes = data.shape.dimensions

    val dataPath = s"$name/values"
    val stepPath = s"$name/steps"

    // 1. Ensure both arrays exist
    if !root.as[py.Dynamic].__contains__(dataPath).as[Boolean] then
      // Data Array
      root.as[py.Dynamic].create_dataset(
        name = dataPath,
        shape = (1 +: dimSizes).toPythonCopy,
        chunks = (1 +: dimSizes).toPythonCopy,
        dtype = "f4",
        fill_value = Float.NaN
      )
      // Steps Array (1D Int64)
      root.as[py.Dynamic].create_dataset(
        name = stepPath,
        shape = List(1).toPythonCopy,
        chunks = List(1).toPythonCopy,
        dtype = "i8",
        fill_value = -1
      )
    else
      assert(
        root.bracketAccess(dataPath).shape.as[Seq[Int]].toList.tail == dimSizes,
        s"Shape mismatch for $dataPath, expected ${dimSizes}, found ${root.bracketAccess(dataPath).shape.as[Seq[Int]].toList.tail}"
      )

    val dSet = root.bracketAccess(dataPath)
    val sSet = root.bracketAccess(stepPath)

    val group = root.bracketAccess(name)
    val nextIdx =
      if group.attrs.__contains__("count").as[Boolean]
      then group.attrs.bracketAccess("count").as[Int]
      else 0

    // 3. Resize both if needed
    if nextIdx >= dSet.shape.bracketAccess(0).as[Int] then
      val newSize = nextIdx + 1
      dSet.resize((newSize +: dimSizes).toPythonCopy)
      sSet.resize(Seq(newSize).toPythonCopy)

    // 4. Write Data and the Step index
    dSet.bracketUpdate(nextIdx, pyData)
    sSet.bracketUpdate(nextIdx, iteration)

    // 5. Update the count attribute
    group.attrs.bracketUpdate("count", nextIdx + 1)

  private def loadPyData(name: String, iteration: Int): Option[py.Dynamic] =
    Option.when(root.as[py.Dynamic].__contains__(s"$name/values").as[Boolean]):
      root.bracketAccess(s"$name/values").bracketAccess(iteration)

  private val `:` = py.Dynamic.global.slice(py.None)

  private def loadPyData(name: String): Option[py.Dynamic] =
    Option.when(root.as[py.Dynamic].__contains__(s"$name/values").as[Boolean]):
      root.bracketAccess(s"$name/values").bracketAccess(`:`)

  private def loadSteps(name: String): Option[Tensor1[Iteration, Int]] =
    Option.when(root.as[py.Dynamic].__contains__(s"$name/steps").as[Boolean]):
      val steps = root.bracketAccess(s"$name/steps").bracketAccess(`:`)
      liftPyTensor[Tuple1[Iteration], Int](steps)

  def loadTensor0[V](name: String, iteration: Int): Option[Tensor0[V]] =
    loadPyData(name, iteration).map(liftPyTensor[EmptyTuple, V](_))

  def loadTensor0[V](name: String): Option[(Tensor1[Iteration, Int], Tensor1[Iteration, V])] =
    for
      steps <- loadSteps(name)
      values <- loadPyData(name).map(liftPyTensor[Tuple1[Iteration], V](_))
    yield (steps, values)

  def loadTensor1[L: Label, V](ax: Axis[L], name: String, iteration: Int): Option[Tensor1[L, V]] =
    loadPyData(name, iteration).map(liftPyTensor[Tuple1[L], V](_))

  def loadTensor1[L: Label, V](ax: Axis[L], name: String): Option[(Tensor1[Iteration, Int], Tensor2[Iteration, L, V])] =
    for
      steps <- loadSteps(name)
      values <- loadPyData(name).map(liftPyTensor[Tuple2[Iteration, L], V](_))
    yield (steps, values)

  def loadTensor2[L1: Label, L2: Label, V](ax1: Axis[L1], ax2: Axis[L2], name: String, iteration: Int): Option[Tensor2[L1, L2, V]] =
    loadPyData(name, iteration).map(liftPyTensor[(L1, L2), V](_))

  def loadTensor3[L1: Label, L2: Label, L3: Label, V](ax1: Axis[L1], ax2: Axis[L2], ax3: Axis[L3], name: String, iteration: Int): Option[Tensor3[L1, L2, L3, V]] =
    loadPyData(name, iteration).map(liftPyTensor[(L1, L2, L3), V](_))

  def loadTensor4[L1: Label, L2: Label, L3: Label, L4: Label, V](ax1: Axis[L1], ax2: Axis[L2], ax3: Axis[L3], ax4: Axis[L4], name: String, iteration: Int): Option[Tensor[(L1, L2, L3, L4), V]] =
    loadPyData(name, iteration).map(liftPyTensor[(L1, L2, L3, L4), V](_))

  def loadTree[Data: TensorTree](name: String, iteration: Int): Option[Data] =
    Option.when(root.as[py.Dynamic].__contains__(name).as[Boolean]):
      val pyData = root.bracketAccess(name)
      TensorTree[Data].fill([T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (path: String) =>
            val raw = pyData.bracketAccess(path).bracketAccess("values").bracketAccess(iteration)
            liftPyTensor[T, V](raw)
      )

@main
def testerino() =
  val logger = new TenZarrLogger("mlp_experiment.zarr")
  // val params = logger.loadTree[MLPParams]("train/params", 0)
  val (steps, losses) = logger.loadTensor0("train/loss").get
  println(losses.shape)

  import plotly.*
  import plotly.element.*
  import plotly.layout.*

  import me.shadaj.scalapy.py
  import me.shadaj.scalapy.py.SeqConverters
  import plotly.Plotly.*

  val currentStep = -1
  val stepIndex = 1

  val params = logger.loadTree[MLPParams]("train/params", stepIndex).get

  // Assuming weight is a 2D Seq[Seq[Double]] or Array[Array[Double]]
  val wMatrix = params.hiddenLayer.weight
  val wFlat = wMatrix.flatten

  val plotySeqwMatrix = toPyTensor(wMatrix).as[Seq[Seq[Double]]]
  val plotySeqwFlat = toPyTensor(wFlat).as[Seq[Double]]

  // Trace 1: Heatmap (Assigned to Axis 1 - Left Side)
  val heatmap = Heatmap(plotySeqwMatrix)
    .withColorscale(ColorScale.NamedScale("Viridis"))

  // Trace 2: Histogram (Assigned to Axis 2 - Right Side)
  val histogram = Histogram(x = plotySeqwFlat)

  // 3. Layout: Define the 1x2 Subplot Grid using Domains
  val layout = Layout()
    .withTitle(s"Layer Weights at Step $currentStep")
    .withShowlegend(false)
    .withWidth(950)
    .withHeight(450)
    .withGrid(
      Grid()
        .withRows(1)
        .withColumns(2)
        .withPattern(Pattern.Independent)
    )

  // 4. Render the plot
  plot("plot.html", Seq(heatmap, histogram), layout)

@main
def testerino2() =
  val logger = new TenZarrLogger("mlp_experiment.zarr")

  import me.shadaj.scalapy.py
  import me.shadaj.scalapy.py.SeqConverters

  // 1. Load the data (exactly as you did before)
  val (stepsRaw, lossesRaw) = logger.loadTensor0("train/loss").get
  val steps = toPyTensor(stepsRaw).as[Seq[Int]]
  val losses = toPyTensor(lossesRaw).as[Seq[Double]]

  // 2. Import matplotlib.pyplot via ScalaPy
  val plt = py.module("matplotlib.pyplot")

  // 3. Create the plot
  // .toPythonCopy converts the Scala Seq into a Python List so matplotlib can read it
  plt.plot(steps.toPythonCopy, losses.toPythonCopy, label = "Training Loss", color = "blue")

  // 4. Set the layout and titles
  plt.title("Training Loss")
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.legend()

  // 5. Display the plot
  plt.show()
