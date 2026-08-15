package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.activation.softmax
import deepwit.base.LinearLayer
import dimwit.Label as Λ

/** Represents an attention mechanism from a target sequence of embeddings onto a source sequence of embeddings.
  *
  * The queries are projected from the target sequence, the keys and values from the source sequence.
  *
  * @tparam Source The axis label for the source sequence.
  * @tparam SourceEmbedding The axis label for the source embedding space.
  * @tparam Target The axis label for the target sequence.
  * @tparam TargetEmbedding The axis label for the target embedding space.
  * @tparam Query The axis label for the query space.
  * @tparam Key The axis label for the key space.
  * @tparam Value The axis label for the value space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param createAttentionMask A function generating a boolean mask to prevent attention to certain positions.
  * @param calculateAttentionScores A function that computes the raw, unmasked attention scores from queries and keys.
  */
class Attention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
    calculateAttentionScores: (Tensor2[Target, Query, V], Tensor2[Source, Key, V]) => Tensor2[Target, Source, V]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, TargetEmbedding, V]) => Tensor2[Target, Value, V]):

  private val projectQuery = LinearLayer(params.queryWeights)
  private val projectKey = LinearLayer(params.keyWeights)
  private val projectValue = LinearLayer(params.valueWeights)

  override def apply(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): Tensor2[Target, Value, V] =
    val queries = target.vmap(Axis[Target])(projectQuery)
    val keys = source.vmap(Axis[Source])(projectKey)
    val values = source.vmap(Axis[Source])(projectValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    attentionWeights.dot(Axis[Source])(values)

  private def calculateAttentionWeights(queries: Tensor2[Target, Query, V], keys: Tensor2[Source, Key, V]) =
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionMask = createAttentionMask(attentionScores.shape)
    val maskedScores = where(attentionMask, attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
    maskedScores.vapply(Axis[Source])(softmax)

object Attention:

  /** Defaults the attention scores to [[scaledDotProductAttentionScores]]. */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      params: Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
      createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new Attention(params, createAttentionMask, scaledDotProductAttentionScores)

  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      params: Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
      createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
      calculateAttentionScores: (Tensor2[Target, Query, V], Tensor2[Source, Key, V]) => Tensor2[Target, Source, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new Attention(params, createAttentionMask, calculateAttentionScores)

  def scaledDotProductAttentionScores[Target: Λ, Source: Λ, Query: Λ, Key: Λ, V: IsFloating](
      queries: Tensor2[Target, Query, V],
      keys: Tensor2[Source, Key, V]
  ): Tensor2[Target, Source, V] =
    val dk = Math.sqrt(keys.shape(Axis[Key]))
    queries.dot(Axis[Query] -> Axis[Key])(keys) /! dk

  case class Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V](
      queryWeights: LinearLayer.Params[TargetEmbedding, Query, V],
      keyWeights: LinearLayer.Params[SourceEmbedding, Key, V],
      valueWeights: LinearLayer.Params[SourceEmbedding, Value, V]
  )

  object Params:

    def apply[SourceEmbedding, TargetEmbedding, Query, Key, Value, V: IsFloating](wq: Tensor2[TargetEmbedding, Query, V], wk: Tensor2[SourceEmbedding, Key, V], wv: Tensor2[SourceEmbedding, Value, V]): Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V] =
      new Params(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    // `Key` names the attention key space here, so the random key needs its qualified type.
    def init[SourceEmbedding: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](queryExtent: AxisExtent[Query], keyExtent: AxisExtent[Key], valueExtent: AxisExtent[Value], sourceEmbeddingExtent: AxisExtent[SourceEmbedding], targetEmbeddingExtent: AxisExtent[TargetEmbedding], vtype: VType[V], key: Random.Key): Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      Params(
        queryWeights = LinearLayer.Params.xavierUniform(targetEmbeddingExtent, queryExtent, vtype, queryKey),
        keyWeights = LinearLayer.Params.xavierUniform(sourceEmbeddingExtent, keyExtent, vtype, keyKey),
        valueWeights = LinearLayer.Params.xavierUniform(sourceEmbeddingExtent, valueExtent, vtype, valueKey)
      )
