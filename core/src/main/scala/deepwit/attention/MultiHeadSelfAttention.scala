package deepwit.attention

import dimwit.*
import deepwit.base.AffineLayer
import dimwit.Label as Λ

/** Represents multi-head self-attention, i.e. multi-head attention of a sequence onto itself. */
abstract class MultiHeadSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    multiHeadAttention: MultiHeadAttention[Context, Embedding, Context, Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  /** Applies the multi-head self-attention to the given context tensor. */
  final def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] = multiHeadAttention(context, context)

  final def applyWithIntermediates(context: Tensor2[Context, Embedding, V]): (Tensor2[Context, Embedding, V], MultiHeadAttention.Intermediates[Context, Context, V]) =
    multiHeadAttention.applyWithIntermediates(context, context)

object MultiHeadSelfAttention:

  /** Holds the parameters for the multi-head self-attention mechanism. */
  case class Params[Embedding, V](
      queryWeights: Tensor3[Head, Embedding, HeadQuery, V],
      keyWeights: Tensor3[Head, Embedding, HeadKey, V],
      valueWeights: Tensor3[Head, Embedding, HeadValue, V],
      outputProjection: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  ):

    /** Converts the self-attention parameters to the corresponding multi-head attention parameters. */
    def asMultiHeadAttentionParams: MultiHeadAttention.Params[Embedding, Embedding, V] =
      MultiHeadAttention.Params(queryWeights, keyWeights, valueWeights, outputProjection)

  object Params:

    def init[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, key, vtype)

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      require(embeddingExtent.size % numHeads == 0)
      import MultiHeadAttention.Params.{xavierUniformHeads, xavierUniformOutputProjection}
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val headExtent = Axis[Head] -> numHeads
      val headQueryExtent = Axis[HeadQuery] -> embeddingExtent.size / numHeads
      val headKeyExtent = Axis[HeadKey] -> embeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> embeddingExtent.size / numHeads
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headQueryExtent, queryKey, vtype),
        keyWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headKeyExtent, keyKey, vtype),
        valueWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headValueExtent, valueKey, vtype),
        outputProjection = xavierUniformOutputProjection(numTransformerLayers, headExtent * headValueExtent, embeddingExtent, projectionKey, vtype)
      )

/** Multi-head self-attention with [[MultiHeadFullAttention]]. */
class MultiHeadFullSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadFullAttention(contextAxis, contextAxis, params.asMultiHeadAttentionParams))

/** Multi-head self-attention with [[MultiHeadCausalAttention]]. */
class MultiHeadCausalSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadCausalAttention(contextAxis, contextAxis, params.asMultiHeadAttentionParams))

/** Multi-head self-attention with [[MultiHeadCustomAttention]]. */
class MultiHeadCustomSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V],
    mask: Shape2[Context, Context] => Tensor2[Context, Context, Bool],
    attentionScore: AttentionScore[Context, Context, HeadQuery, HeadKey, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadCustomAttention(params.asMultiHeadAttentionParams, mask, attentionScore))
