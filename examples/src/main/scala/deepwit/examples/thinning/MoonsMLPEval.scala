package deepwit.examples.thinning

import dimwit.*

import deepwit.examples.dataset.TwoMoons
import deepwit.examples.dataset.TwoMoons.{Feature, Output}
import dimwit.Conversions.given
import io.circe.Json
import plotwit.*

import deepwit.activation.softmax
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.examples.newestRun

import plotwit.PlotTargets.desktopBrowser

/** Thins at inference, where most libraries switch their dropout off.
  *
  * `params.thin(probability, key)` is another parameter set, and a key names which one. Asking a
  * hundred of the resulting models the same question and looking at how much they disagree is all
  * Monte Carlo dropout is — here drawn over the whole plane at once.
  *
  * The moons themselves are covered in training data, so the thinned models agree there. Between and
  * beyond them there is nothing to have pinned the model down, and they come apart.
  *
  * @param givenPath The directory a training run wrote its checkpoints to. Left out, the most recent
  *                  run is read.
  */
@main
def eval(givenPath: String*): Unit =

  val numSamples = 400
  val noiseScale = 0.15f

  val thinningProbability = 0.2f
  val numDraws = 100
  val gridSize = 300
  val tilesPerSide = 10

  val checkpointPath = givenPath.headOption.getOrElse(newestRun("out/MoonsMLP"))
  val checkpointer = TensorTreeCheckpointer(checkpointPath)
  val iterations = checkpointer.iterations
  require(iterations.nonEmpty, s"No checkpoint to render in $checkpointPath.")

  val iteration = iterations.max
  val state = checkpointer.load[TrainState](iteration).get
  println(f"Loaded $checkpointPath after $iteration steps, at cost ${state.lastCost.item}%.6f.")

  val rowExtent = Axis[Row] -> gridSize
  val columnExtent = Axis[Column] -> gridSize
  val grid = Shape(rowExtent, columnExtent)
  val coordinates = TwoMoons.grid(rowExtent, columnExtent, lowerLeft = (-2f, -1.5f), upperRight = (3f, 2f))

  // The model knows nothing of any of this, so one traced graph serves the deployed parameters
  // and every thinning of them alike.
  val sweep = jit: (params: MoonsMLP.Params) =>
    coordinates
      .vmap(Axis[Row]):
        _.vmap(Axis[Column]): point =>
          softmax(MoonsMLP(params).logits(point)).slice(Axis[Output].at(1))

  val deployed = sweep(state.params)
  val draws = Key(7).split(numDraws).toList.map(key => sweep(state.params.thin(thinningProbability, key)))
  val ensemble = stack(draws, Axis[Draw])
  val disagreement = ensemble.std(Axis[Draw])

  println(f"Disagreement over the window: mean ${disagreement.mean.item}%.4f, at most ${disagreement.max.item}%.4f of a possible $mostTwoModelsCanDiffer%.1f.")

  display(hconcat(List(
    trainingPlot(TwoMoons.sampleFix(numSamples, noiseScale)),
    probabilityPlot(deployed, "Stored parameters\n(unthinned)"),
    probabilityPlot(ensemble.mean(Axis[Draw]), f"Mean of $numDraws thinned models"),
    disagreementPlot(disagreement, f"Disagreement across $numDraws thinnings\n(predictive standard deviation)"),
    probabilityPlot(tiled(draws.map(shrunk(_, tilesPerSide)), tilesPerSide), f"Each of the $numDraws thinned models")
  )))

private def trainingPlot(dataset: TwoMoons.Dataset): VegaLiteSpec =
  plots.scatterPlot(
    dataset.features.slice(Axis[Feature].at(0)),
    dataset.features.slice(Axis[Feature].at(1)),
    dataset.labels.toArray.map(moon => s"moon $moon").toSeq,
    _.title := "The data the model was fitted to",
    _.encoding.x.title := "x",
    _.encoding.y.title := "y",
    _.encoding.size := Json.obj("value" -> Json.fromInt(18))
  )

private def probabilityPlot[R: Label, C: Label](field: Tensor2[R, C, Float32], title: String): VegaLiteSpec = fieldPlot(field, title, moonColours)

private def disagreementPlot[R: Label, C: Label](field: Tensor2[R, C, Float32], title: String): VegaLiteSpec = fieldPlot(field, title, difference => agreementColours(difference / mostTwoModelsCanDiffer))

private def fieldPlot[R: Label, C: Label](field: Tensor2[R, C, Float32], title: String, colours: Float => Colour): VegaLiteSpec =
  plots.imagePlot(
    painted(field, colours),
    _.title := title,
    _.mark.width := panelSize,
    _.mark.height := panelSize,
    _.mark.smooth := true
  )

private def painted[R: Label, C: Label](field: Tensor2[R, C, Float32], colours: Float => Colour): Tensor3[C, R, Channel, UInt8] =
  val rows = field.shape(Axis[R])
  val columns = field.shape(Axis[C])
  val values = field.toArray
  // Written as bytes and read back as UInt8, so intensities above 127 keep their bit pattern.
  val pixels = new Array[Byte](columns * rows * 3)
  for row <- 0 until rows; column <- 0 until columns do
    val (red, green, blue) = colours(values(row)(column))
    val at = (column * rows + row) * 3
    pixels(at) = red.toByte
    pixels(at + 1) = green.toByte
    pixels(at + 2) = blue.toByte
  Tensor(Shape(Axis[C] -> columns, Axis[R] -> rows, Axis[Channel] -> 3), VType[UInt8]).fromArray(pixels)

private def shrunk(field: Tensor2[Row, Column, Float32], factor: Int): Tensor2[Row, Column, Float32] =
  def kept(extent: Int) = (0 until extent by factor).toIndexedSeq
  field
    .slice(Axis[Row].at(kept(field.shape(Axis[Row]))))
    .slice(Axis[Column].at(kept(field.shape(Axis[Column]))))

/** Lays the fields out as a `perSide` x `perSide` tiling, as one field over a larger plane.
  *
  * The axis they are stacked along is relabelled into a pair of axes and each folded into the grid
  * axis beside it, so the tiles meet exactly: the gaps a plotting library leaves between separate
  * panels never exist to be closed.
  */
private def tiled(fields: Seq[Tensor2[Row, Column, Float32]], perSide: Int): Tensor2[Prime[Row] |*| Row, Prime[Column] |*| Column, Float32] =
  require(fields.size == perSide * perSide, s"A $perSide x $perSide tiling needs ${perSide * perSide} fields, but was given ${fields.size}.")
  stack(fields, Axis[Draw])
    .relabel(Axis[Draw].as(Axis[Prime[Row] |*| Prime[Column]]))
    .rearrange(
      (Axis[Prime[Row] |*| Row], Axis[Prime[Column] |*| Column]),
      (
        Axis[Prime[Row]] -> perSide,
        Axis[Prime[Column]] -> perSide,
        Axis[Row] -> fields.head.shape(Axis[Row]),
        Axis[Column] -> fields.head.shape(Axis[Column])
      )
    )

private type Colour = (Int, Int, Int)

private val moonColours: Float => Colour = ramp((49, 88, 145), (247, 245, 240), (200, 96, 32))
private val agreementColours: Float => Colour = ramp((12, 8, 30), (155, 40, 90), (250, 235, 160))

private def ramp(low: Colour, middle: Colour, high: Colour)(value: Float): Colour =
  val clamped = math.min(1f, math.max(0f, value))
  if clamped < 0.5f then between(low, middle, clamped * 2f)
  else between(middle, high, (clamped - 0.5f) * 2f)

private def between(from: Colour, to: Colour, at: Float): Colour =
  def mix(a: Int, b: Int) = math.round(a + (b - a) * at)
  (mix(from._1, to._1), mix(from._2, to._2), mix(from._3, to._3))

/** Two models can differ by at most a half in the probability they give one point. */
private val mostTwoModelsCanDiffer = 0.5f

private val panelSize = 200
