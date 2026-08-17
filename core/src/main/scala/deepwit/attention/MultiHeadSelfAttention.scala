package deepwit.attention

import dimwit.*
import deepwit.base.AffineLayer
import dimwit.Label as Λ

/** Represents multi-head self-attention, i.e. multi-head attention of a sequence onto itself.
  *
  * Which positions a position may attend to is left to the implementation:
  * [[MultiHeadFullSelfAttention]], [[MultiHeadCausalSelfAttention]] or [[MultiHeadCustomSelfAttention]].
  *
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param multiHeadAttention The attention the sequence runs onto itself.
  */
abstract class MultiHeadSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    multiHeadAttention: MultiHeadAttention[Context, Embedding, Context, Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    multiHeadAttention(context, context)

/** Multi-head self-attention where every position may attend to every other one.
  *
  * @param contextAxis The axis of the sequence attending onto itself.
  */
class MultiHeadFullSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](
      MultiHeadFullAttention(contextAxis, contextAxis, params.asMultiHeadAttentionParams)
    )

/** Multi-head self-attention where a position may only attend to positions up to its own index.
  *
  * @param contextAxis The axis of the sequence attending onto itself.
  */
class MultiHeadCausalSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](
      MultiHeadCausalAttention(contextAxis, contextAxis, params.asMultiHeadAttentionParams)
    )

/** Multi-head self-attention restricted by a caller-supplied mask.
  *
  * @param mask A function generating a boolean mask to prevent attention to certain positions.
  */
class MultiHeadCustomSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V],
    mask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends MultiHeadSelfAttention[Context, Embedding, V](
      MultiHeadCustomAttention(params.asMultiHeadAttentionParams, mask)
    )

object MultiHeadSelfAttention:

  case class Params[Embedding, V](
      queryWeights: Tensor3[Head, Embedding, HeadQuery, V],
      keyWeights: Tensor3[Head, Embedding, HeadKey, V],
      valueWeights: Tensor3[Head, Embedding, HeadValue, V],
      outputProjection: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  ):

    /** The same parameters read as attention of the sequence onto itself. */
    def asMultiHeadAttentionParams: MultiHeadAttention.Params[Embedding, Embedding, V] =
      MultiHeadAttention.Params(queryWeights, keyWeights, valueWeights, outputProjection)

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Key): Params[Embedding, V] =
      require(embeddingExtent.size % numHeads == 0)
      import MultiHeadAttention.Params.{xavierUniformHeads, xavierUniformOutputProjection}
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val headExtent = Axis[Head] -> numHeads
      val headQueryExtent = Axis[HeadQuery] -> embeddingExtent.size / numHeads
      val headKeyExtent = Axis[HeadKey] -> embeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> embeddingExtent.size / numHeads
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headQueryExtent, vtype, queryKey),
        keyWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headKeyExtent, vtype, keyKey),
        valueWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headValueExtent, vtype, valueKey),
        outputProjection = xavierUniformOutputProjection(numTransformerLayers, headExtent * headValueExtent, embeddingExtent, vtype, projectionKey)
      )
