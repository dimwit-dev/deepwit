package deepwit.embedder

import dimwit.*
import deepwit.cnn.AffineConv2DLayer
import deepwit.embedder.PositionalEncoding.sinusoidal2D
import dimwit.Label as Λ

/** Cuts an image into non-overlapping patches and embeds every patch into a sequence element.
  *
  * The patches are produced by a strided convolution whose stride equals the kernel size, and are
  * enriched with a 2D sinusoidal positional encoding before being flattened into a sequence.
  *
  * @tparam Width The axis label for the image width.
  * @tparam Height The axis label for the image height.
  * @tparam Channel The axis label for the image channels.
  * @tparam PatchEmbedding The axis label for the patch embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  */
class ImageToPatchEmbedder[
    Width: Λ,
    Height: Λ,
    Channel: Λ,
    PatchEmbedding: Λ,
    V: IsFloating
](
    params: ImageToPatchEmbedder.Params[Width, Height, Channel, PatchEmbedding, V]
) extends (Tensor3[Width, Height, Channel, V] => Tensor2[Width |*| Height, PatchEmbedding, V]):

  private val convLayer =
    val kernelShape = params.conv.kernel.shape
    val kernelSize = (kernelShape.extent(Axis[Width]), kernelShape.extent(Axis[Height]))
    AffineConv2DLayer(params.conv, stride = kernelSize)

  override def apply(img: Tensor3[Width, Height, Channel, V]): Tensor2[Width |*| Height, PatchEmbedding, V] =
    val patches = convLayer(img)
    val patchesPos = patches + sinusoidal2D(patches.shape)
    patchesPos.flatten((Axis[Width], Axis[Height]))

object ImageToPatchEmbedder:

  case class Params[PatchWidth, PatchHeight, Channel, PatchEmbedding, V](
      conv: AffineConv2DLayer.Params[PatchWidth, PatchHeight, Channel, PatchEmbedding, V]
  )

  object Params:

    def xavierUniform[PatchWidth: Λ, PatchHeight: Λ, Channel: Λ, PatchEmbedding: Λ, V: IsFloating](patchWidthExtent: AxisExtent[PatchWidth], patchHeightExtent: AxisExtent[PatchHeight], channelExtent: AxisExtent[Channel], embeddingExtent: AxisExtent[PatchEmbedding], vtype: VType[V], key: Key): Params[PatchWidth, PatchHeight, Channel, PatchEmbedding, V] =
      Params(conv = AffineConv2DLayer.Params.xavierUniform(patchWidthExtent, patchHeightExtent, channelExtent, embeddingExtent, vtype, key))
