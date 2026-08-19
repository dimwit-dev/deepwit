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
      multiHeadAttention: MultiHeadAttention.Params[Embedding, Embedding, V]
  )

  object Params:

    def init[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      Params(MultiHeadAttention.Params.init(numTransformerLayers, numHeads, embeddingExtent, embeddingExtent, key, vtype))

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      Params(MultiHeadAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, embeddingExtent, key, vtype))

/** Multi-head self-attention with [[MultiHeadFullAttention]]. */
class MultiHeadFullSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadFullAttention(contextAxis, contextAxis, params.multiHeadAttention))

/** Multi-head self-attention with [[MultiHeadCausalAttention]]. */
class MultiHeadCausalSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadCausalAttention(contextAxis, contextAxis, params.multiHeadAttention))

/** Multi-head self-attention with [[MultiHeadCustomAttention]]. */
class MultiHeadCustomSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V],
    mask: Shape2[Context, Context] => Tensor2[Context, Context, Bool],
    attentionScore: AttentionScore[Context, Context, HeadQuery, HeadKey, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadCustomAttention(params.multiHeadAttention, mask, attentionScore))
