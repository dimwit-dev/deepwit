package deepwit.embedder

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

object PositionalEncoding:

  def sinusoidal2D[X: Λ, Y: Λ, Embedding: Λ, V: IsFloating](shape: Shape3[X, Y, Embedding]): Tensor3[X, Y, Embedding, V] =
    // 1. Prepare things we need for positional encoding
    val xExtent = shape.extent(Axis[X]).size
    val yExtent = shape.extent(Axis[Y]).size
    val embedDim = shape.extent(Axis[Embedding]).size

    // Each spatial dimension (x, y) gets exactly half the embedding capacity (embedDim / 2).
    // Since we generate both sine and cosine pairs for each scale, we divide by 2 again.
    // Therefore, the number of unique frequency scales needed is (embedDim / 2) / 2 = embedDim / 4.
    require(embedDim % 4 == 0, s"Embedding dimension ($embedDim) must be cleanly divisible by 4 to generate symmetrical 2D sinusoidal positional encodings.")
    val scaleCount = embedDim / 4
    val posScales = (Tensor1(Axis[Embedding]).fromArray(Array.range(0, scaleCount)).asFloat(VType[V]) *! -(Tensor0(VType[V])(10000.0f).log / scaleCount)).exp

    // 2. Prepare X-axis
    val xPosRaw = Tensor1(Axis[X]).fromArray(Array.range(0, xExtent))
    val xPosScaled = xPosRaw.asFloat(VType[V]).vmap(Axis[X])(_ *! posScales)
    val xPosEncoded = concatenate(xPosScaled.sin, xPosScaled.cos, concatAxis = Axis[Embedding])

    // 3. Prepare Y-axis
    val yPosRaw = Tensor1(Axis[Y]).fromArray(Array.range(0, yExtent))
    val yPosScaled = yPosRaw.asFloat(VType[V]).vmap(Axis[Y])(_ *! posScales)
    val yPosEncoded = concatenate(yPosScaled.sin, yPosScaled.cos, concatAxis = Axis[Embedding])

    // 4. Expansion into 2D Grids and Concatenation
    val xPosGrid = stack(List.fill(yExtent)(xPosEncoded), newAxis = Axis[Y], afterAxis = Axis[X])
    val yPosGrid = stack(List.fill(xExtent)(yPosEncoded), newAxis = Axis[X])

    concatenate(xPosGrid, yPosGrid, concatAxis = Axis[Embedding])
