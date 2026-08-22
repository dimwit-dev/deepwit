package deepwit.examples.dataset

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal

/** Two interleaving half circles, the smallest dataset no straight line can separate.
  *
  * Small enough to train in seconds and two-dimensional, so a model fitted to it can be asked about
  * every point of the plane at once and the answer drawn as a picture.
  */
object TwoMoons:

  trait Sample derives Label

  /** The plane a point is drawn from. */
  trait Feature derives Label

  /** Which moon a point came from. */
  trait Output derives Label

  val featureExtent: AxisExtent[Feature] = Axis[Feature] -> 2
  val outputExtent: AxisExtent[Output] = Axis[Output] -> 2

  case class BatchSample[Batch](features: Tensor2[Batch, Feature, Float32], labels: Tensor1[Batch, Int32])

  case class Dataset(features: Tensor2[Sample, Feature, Float32], labels: Tensor1[Sample, Int32]):

    def toBatchStream[Batch: Label](batchExtent: AxisExtent[Batch]): Iterator[BatchSample[Batch]] =
      val totalSamples = features.shape(Axis[Sample])
      val batchSize = batchExtent.size
      Iterator.iterate(0)(_ + batchSize).map: offset =>
        val batchIds = (0 until batchSize).map(i => (offset + i) % totalSamples)
        val batchFeatures = features.slice(Axis[Sample].at(batchIds)).relabel(Axis[Sample], Axis[Batch])
        val batchLabels = labels.slice(Axis[Sample].at(batchIds)).relabel(Axis[Sample], Axis[Batch])
        BatchSample(batchFeatures, batchLabels)

  def sampleFix(count: Int, noiseScale: Float): Dataset = sample(count, noiseScale, Key(42))

  /** Draws `count` points, half from each moon, blurred by Gaussian noise.
    *
    * Each moon is a half turn swept at a constant rate, the lower one shifted so that the two
    * interleave. The noise is what leaves the boundary between them genuinely uncertain rather than
    * merely curved.
    */
  def sample(count: Int, noiseScale: Float, key: Key): Dataset =
    require(count >= 4, s"Two moons need at least two points each, but was asked for $count.")
    val upperCount = count / 2
    val sampleExtent = Axis[Sample] -> count

    val coordinates = new Array[Float](count * 2)
    val moons = new Array[Int](count)
    for sample <- 0 until count do
      val onUpper = sample < upperCount
      val within = if onUpper then sample else sample - upperCount
      val alongMoon = if onUpper then upperCount else count - upperCount
      val angle = math.Pi * within / (alongMoon - 1)
      val (x, y) =
        if onUpper then (math.cos(angle).toFloat, math.sin(angle).toFloat)
        else (1f - math.cos(angle).toFloat, 0.5f - math.sin(angle).toFloat)
      coordinates(sample * 2) = x
      coordinates(sample * 2 + 1) = y
      moons(sample) = if onUpper then 0 else 1

    val clean = Tensor(Shape(sampleExtent, featureExtent), VType[Float32]).fromArray(coordinates)
    val noise = Normal.standardNormal(clean.shape).sample(key) *! noiseScale
    Dataset(clean + noise, Tensor(Shape1(sampleExtent), VType[Int32]).fromArray(moons))

  /** The plane, sampled on a `rowExtent` x `columnExtent` grid, one position per row of the result.
    *
    * Laid out along a single axis because that is what the network reads: it takes one position at a
    * time, and knows nothing of the grid the positions were taken from.
    *
    * Rows run from the top of the window down, which is the order a heatmap draws them in, so that
    * a field rendered from this grid is the right way up.
    */
  def grid[Row: Label, Column: Label](
      rowExtent: AxisExtent[Row],
      columnExtent: AxisExtent[Column],
      lowerLeft: (Float, Float),
      upperRight: (Float, Float)
  ): Tensor3[Row, Column, Feature, Float32] =
    val (left, bottom) = lowerLeft
    val (right, top) = upperRight
    val rows = rowExtent.size
    val columns = columnExtent.size
    val coordinates = new Array[Float](rows * columns * 2)
    for row <- 0 until rows; column <- 0 until columns do
      val at = (row * columns + column) * 2
      coordinates(at) = left + (right - left) * column / (columns - 1)
      coordinates(at + 1) = top - (top - bottom) * row / (rows - 1)
    Tensor(Shape(rowExtent, columnExtent, featureExtent), VType[Float32])
      .fromArray(coordinates)
