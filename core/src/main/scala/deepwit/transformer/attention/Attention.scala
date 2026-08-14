package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.softmax
import deepwit.base.LinearLayer
import dimwit.Label as Λ

/** Represents an attention mechanism from a target sequence of embeddings onto a source sequence of embeddings.
  *
  * @tparam Source The axis label for the source sequence.
  * @tparam SourceEmbedding The axis label for the source embedding space.
  * @tparam Target The axis label for the target sequence.
  * @tparam Embedding The axis label for the target embedding space.
  * @tparam Query The axis label for the query space.
  * @tparam Key The axis label for the key space.
  * @tparam Value The axis label for the value space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param createAttentionMask A function generating a boolean mask to prevent attention to certain positions.
  * @param calculateAttentionScores A function that computes the raw, unmasked attention scores from queries and keys.
  */
case class Attention[Source: Λ, SourceEmbedding: Λ, Target: Λ, Embedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: Attention.Params[SourceEmbedding, Embedding, Query, Key, Value, V],
    createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
    calculateAttentionScores: (Tensor2[Target, Query, V], Tensor2[Source, Key, V]) => Tensor2[Target, Source, V]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, Embedding, V]) => Tensor2[Target, Value, V]):

  private val projectQuery = LinearLayer(params.queryWeights)
  private val projectKey = LinearLayer(params.keyWeights)
  private val projectValue = LinearLayer(params.valueWeights)

  override def apply(crossContext: Tensor2[Source, SourceEmbedding, V], context: Tensor2[Target, Embedding, V]): Tensor2[Target, Value, V] =
    val queries = context.vmap(Axis[Target])(projectQuery)
    val keys = crossContext.vmap(Axis[Source])(projectKey)
    val values = crossContext.vmap(Axis[Source])(projectValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    attentionWeights.dot(Axis[Source])(values)

  private def calculateAttentionWeights(queries: Tensor2[Target, Query, V], keys: Tensor2[Source, Key, V]) =
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionMask = createAttentionMask(attentionScores.shape)
    val attentionWeights = where(attentionMask, attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
    attentionWeights.vapply(Axis[Source])(softmax)

object Attention:

  /** Default to Scaled Dot-Product Attention ([[scaledDotProductAttentionScores]]). */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, Embedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      params: Params[SourceEmbedding, Embedding, Query, Key, Value, V],
      createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool]
  ): Attention[Source, SourceEmbedding, Target, Embedding, Query, Key, Value, V] =
    new Attention(params, createAttentionMask, scaledDotProductAttentionScores)

  def scaledDotProductAttentionScores[Target: Λ, Source: Λ, Query: Λ, Key: Λ, V: IsFloating](
      queries: Tensor2[Target, Query, V],
      keys: Tensor2[Source, Key, V]
  ): Tensor2[Target, Source, V] =
    val dk = Math.sqrt(keys.shape(Axis[Key]))
    queries.dot(Axis[Query] -> Axis[Key])(keys) /! dk

  case class Params[SourceEmbedding, Embedding, Query, Key, Value, V: IsFloating](
      queryWeights: LinearLayer.Params[Embedding, Query, V],
      keyWeights: LinearLayer.Params[SourceEmbedding, Key, V],
      valueWeights: LinearLayer.Params[SourceEmbedding, Value, V]
  )

  object Params:

    def apply[SourceEmbedding, Embedding, Query, Key, Value, V: IsFloating](wq: Tensor2[Embedding, Query, V], wk: Tensor2[SourceEmbedding, Key, V], wv: Tensor2[SourceEmbedding, Value, V]): Params[SourceEmbedding, Embedding, Query, Key, Value, V] =
      new Params(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[SourceEmbedding: Λ, Embedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](queryExtent: AxisExtent[Query], keyExtent: AxisExtent[Key], valueExtent: AxisExtent[Value], crossEmbeddingExtent: AxisExtent[SourceEmbedding], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[SourceEmbedding, Embedding, Query, Key, Value, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      Params(
        queryWeights = LinearLayer.Params.xavierUniform(embeddingExtent, queryExtent, vtype, queryKey),
        keyWeights = LinearLayer.Params.xavierUniform(crossEmbeddingExtent, keyExtent, vtype, keyKey),
        valueWeights = LinearLayer.Params.xavierUniform(crossEmbeddingExtent, valueExtent, vtype, valueKey)
      )
