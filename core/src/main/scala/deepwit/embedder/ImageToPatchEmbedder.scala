package deepwit.embedder

import dimwit.*
import dimwit.Conversions.given
import deepwit.cnn.AffineConv2DLayer
import dimwit.stats.Normal

case class ConvImageToPatchEmbedder[
    Width: Label,
    Height: Label,
    Channel: Label,
    PatchEmbedding: Label
](
    params: ConvImageToPatchEmbedder.Params[Width, Height, Channel, PatchEmbedding]
) extends (Tensor3[Width, Height, Channel, Float] => Tensor2[Width |*| Height, PatchEmbedding, Float]):

  private val convLayer =
    val kernelShape = params.conv.kernel.shape
    val kernelSize = (kernelShape.extent(Axis[Width]), kernelShape.extent(Axis[Height]))
    AffineConv2DLayer(AffineConv2DLayer.HyperParams(stride = kernelSize))(params.conv)

  override def apply(img: Tensor3[Width, Height, Channel, Float]): Tensor2[Width |*| Height, PatchEmbedding, Float] =
    val patches = encodeToPatches(img)
    val patchesPos = patches + positionalEncoding2D(patches.shape)
    patchesPos.flatten((Axis[Width], Axis[Height]))

  protected def encodeToPatches(img: Tensor[(Width, Height, Channel), Float]): Tensor[(Width, Height, PatchEmbedding), Float] =
    convLayer(img)

  protected def positionalEncoding2D(shape: Shape3[Width, Height, PatchEmbedding]): Tensor3[Width, Height, PatchEmbedding, Float] =
    // 1. Prepare things we need for positional encoding
    val widthExtent = shape.extent(Axis[Width]).size
    val heightExtent = shape.extent(Axis[Height]).size
    val embedDim = shape.extent(Axis[PatchEmbedding]).size
    val scaleCount = embedDim / 4
    val posScales = (Tensor1(Axis[PatchEmbedding]).fromArray(Array.range(0, scaleCount)).asFloat *! -(Tensor0(10000.0f).log / scaleCount)).exp

    // 2. Prepare Width (X-axis)
    val widthPosRaw = Tensor1(Axis[Width]).fromArray(Array.range(0, widthExtent))
    val widthPosScaled = widthPosRaw.asFloat.vmap(Axis[Width])(_ *! posScales)
    val widthPosEncoded = concatenate(widthPosScaled.sin, widthPosScaled.cos, concatAxis = Axis[PatchEmbedding])

    // 3. Prepare Height (Y-axis)
    val heightPosRaw = Tensor1(Axis[Height]).fromArray(Array.range(0, heightExtent))
    val heightPosScaled = heightPosRaw.asFloat.vmap(Axis[Height])(_ *! posScales)
    val heightPosEncoded = concatenate(heightPosScaled.sin, heightPosScaled.cos, concatAxis = Axis[PatchEmbedding])

    // 4. Expansion into 2D Grids and Concatenation
    val widthPosGrid = stack(List.fill(heightExtent)(widthPosEncoded), newAxis = Axis[Height]).transpose(Axis[Width], Axis[Height], Axis[PatchEmbedding])
    val heightPosGrid = stack(List.fill(widthExtent)(heightPosEncoded), newAxis = Axis[Width])

    concatenate(widthPosGrid, heightPosGrid, concatAxis = Axis[PatchEmbedding])

object ConvImageToPatchEmbedder:

  case class Params[PatchWidth, PatchHeight, Channel, PatchEmbedding](
      conv: AffineConv2DLayer.Params[PatchWidth, PatchHeight, Channel, PatchEmbedding]
  )

  object Params:

    def xavierUniform[PatchWidth: Label, PatchHeight: Label, Channel: Label, PatchEmbedding: Label](patchWidthExtent: AxisExtent[PatchWidth], patchHeightExtent: AxisExtent[PatchHeight], channelExtent: AxisExtent[Channel], embeddingExtent: AxisExtent[PatchEmbedding], key: Random.Key): Params[PatchWidth, PatchHeight, Channel, PatchEmbedding] =
      Params(conv = AffineConv2DLayer.Params.xavierUniform(patchWidthExtent, patchHeightExtent, channelExtent, embeddingExtent, key))
