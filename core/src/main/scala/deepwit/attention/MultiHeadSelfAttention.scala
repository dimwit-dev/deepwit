package deepwit.attention

import dimwit.*
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

/** Multi-head self-attention with [[MultiHeadCausalAttention]].
  *
  * Besides attending over a whole context at once, it can decode one position at a time with a [[KVCache]]:
  * [[prefill]] attends over a prompt and keeps its keys and values, and [[decode]] continues from there.
  */
class MultiHeadCausalSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadCausalAttention(contextAxis, contextAxis, params.multiHeadAttention)):

  private val weights = params.multiHeadAttention

  // Decoding attends from one position to the cache, which the fused kernel cannot, so it always attends head by head.
  private val decoding = MultiHeadUnfusedCausalAttention(contextAxis, contextAxis, weights, AttentionScore.scaledDotProduct)

  /** Attends over `context` like [[apply]], and keeps its keys and values to [[decode]] the positions after it. */
  def prefill(context: Tensor2[Context, Embedding, V]): (Tensor2[Context, Embedding, V], KVCache[Context, V]) =
    val (attended, intermediates) = applyWithIntermediates(context)
    (attended, KVCache(intermediates.keys, intermediates.values))

  /** Attends from the `embedding` at `position` to itself and every position before it, as [[apply]] would.
    *
    * @param cache Holds the keys and values of the positions before `position`.
    * @param position May be traced, e.g. under `jit` or in a loop.
    * @return The attended embedding, and `cache` with the key and value of `position` written into it.
    */
  def decode(cache: KVCache[Context, V], position: Tensor0[Int32], embedding: Tensor1[Embedding, V]): (Tensor1[Embedding, V], KVCache[Context, V]) =
    val query = embedding.dot(Axis[Embedding])(weights.queryWeights)
    val key = embedding.dot(Axis[Embedding])(weights.keyWeights)
    val value = embedding.dot(Axis[Embedding])(weights.valueWeights)
    val keys = cache.keys.updated(contextAxis.at(position))(key)
    val values = cache.values.updated(contextAxis.at(position))(value)
    val headValues = decoding.attendAt(
      queries = stack(Seq(query), contextAxis),
      keys = keys,
      values = values,
      targetPositions = stack(Seq(position), contextAxis),
      sourcePositions = Tensor1(contextAxis).fromRange(0 until keys.shape(contextAxis))
    )
    (decoding.projectHeads(headValues).select(contextAxis.at(0)), KVCache(keys, values))

/** Multi-head self-attention with [[MultiHeadCustomAttention]]. */
class MultiHeadCustomSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V],
    mask: AttentionMask[Context, Context],
    attentionScore: AttentionScore[Context, Context, HeadQuery, HeadKey, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](MultiHeadCustomAttention(params.multiHeadAttention, mask, attentionScore))
