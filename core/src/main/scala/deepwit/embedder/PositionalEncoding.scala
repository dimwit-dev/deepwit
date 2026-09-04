package deepwit.embedder

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

object PositionalEncoding:

  /** The axis the frequencies live on, each contributing both a sine and a cosine. */
  private trait Scale derives Label

  /** The default ratio between the fastest and the slowest oscillation, as chosen for sequences of
    * text. A grid far shorter than that leaves the slowest scales all but constant across it.
    */
  val defaultFrequencyRange: Float = 10000f

  /** The positions of a grid of the given extent: 0, 1, ... n - 1. */
  def gridPositions[P: Λ, V: IsFloating](extent: AxisExtent[P], vtype: VType[V]): Tensor1[P, V] =
    Tensor1(Axis[P]).fromArray(Array.range(0, extent.size)).asFloat(vtype)

  /** Encodes each position as sines and cosines of it, at geometrically spaced frequencies, as
    * described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
    *
    * The positions are given rather than derived, so they need not be a grid's indices: any
    * position the encoding is evaluated at means the same thing to a model trained on any other.
    *
    * @param embeddingExtent Must be even, to pair each frequency's sine with its cosine.
    * @param frequencyRange The ratio between the fastest and the slowest oscillation.
    */
  def sinusoidal[P: Λ, Embedding: Λ, V: IsFloating](
      positions: Tensor1[P, V],
      embeddingExtent: AxisExtent[Embedding],
      frequencyRange: Float = defaultFrequencyRange
  ): Tensor2[P, Embedding, V] =
    val embedDim = embeddingExtent.size
    require(embedDim % 2 == 0, s"An embedding dimension must be even to pair each sine with a cosine, but was $embedDim.")
    require(frequencyRange > 1f, s"A frequency range must exceed one, but was $frequencyRange.")
    val vtype = positions.vtype
    val scaleCount = embedDim / 2
    val scales = (Tensor1(Axis[Scale])
      .fromArray(Array.range(0, scaleCount))
      .asFloat(vtype) *! -(Tensor0(vtype)(frequencyRange).log / scaleCount)).exp
    val scaled = positions.vmap(Axis[P])(_ *! scales)
    concatenate(scaled.sin, scaled.cos, concatAxis = Axis[Scale]).relabel(Axis[Scale], Axis[Embedding])

  /** Encodes each point of a plane, giving each of the two axes half of the embedding.
    *
    * @param embeddingExtent Must be divisible by four: each axis takes half, and each half pairs sines with cosines.
    * @param frequencyRange The ratio between the fastest and the slowest oscillation.
    */
  def sinusoidal2D[X: Λ, Y: Λ, Embedding: Λ, V: IsFloating](
      xPositions: Tensor1[X, V],
      yPositions: Tensor1[Y, V],
      embeddingExtent: AxisExtent[Embedding],
      frequencyRange: Float = defaultFrequencyRange
  ): Tensor3[X, Y, Embedding, V] =
    val embedDim = embeddingExtent.size
    require(embedDim % 4 == 0, s"Embedding dimension ($embedDim) must be cleanly divisible by 4 to generate symmetrical 2D sinusoidal positional encodings.")
    val perAxisExtent = Axis[Embedding] -> embedDim / 2
    val xEncoded = sinusoidal(xPositions, perAxisExtent, frequencyRange)
    val yEncoded = sinusoidal(yPositions, perAxisExtent, frequencyRange)

    val xGrid = stack(List.fill(yPositions.shape(Axis[Y]))(xEncoded), newAxis = Axis[Y], afterAxis = Axis[X])
    val yGrid = stack(List.fill(xPositions.shape(Axis[X]))(yEncoded), newAxis = Axis[X])
    concatenate(xGrid, yGrid, concatAxis = Axis[Embedding])

  /** Encodes the points of a grid, at its own indices. */
  def sinusoidal2D[X: Λ, Y: Λ, Embedding: Λ, V: IsFloating](shape: Shape3[X, Y, Embedding]): Tensor3[X, Y, Embedding, V] =
    val vtype = VType[V]
    sinusoidal2D(
      gridPositions(shape.extent(Axis[X]), vtype),
      gridPositions(shape.extent(Axis[Y]), vtype),
      shape.extent(Axis[Embedding])
    )
