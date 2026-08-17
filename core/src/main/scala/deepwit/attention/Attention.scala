package deepwit.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.activation.softmax
import deepwit.base.LinearLayer
import dimwit.Label as Λ

/** Represents an attention mechanism from a target sequence of embeddings onto a source sequence of embeddings.
  *
  * The queries are projected from the target sequence, the keys and values from the source sequence.
  * Which source positions a target position may attend to is left to the implementation: * [[FullAttention]], [[CausalAttention]] or [[CustomAttention]].
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
  * @param attentionScore Computes the raw, unmasked attention scores from the queries and keys.
  */
abstract class Attention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, TargetEmbedding, V]) => Tensor2[Target, Value, V]):

  /** The source positions that each target position may attend to. */
  protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool]

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
    val attentionScores = attentionScore(queries, keys)
    val attentionMask = createAttentionMask(attentionScores.shape)
    val maskedScores = where(attentionMask, attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
    maskedScores.vapply(Axis[Source])(softmax)

object Attention:

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

/** Attention where every target position may attend to every source position.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
class FullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool] = fullMask(scoreShape)

object FullAttention:

  /** Defaults the attention scores to [[ScaledDotProduct]]. */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V]
  ): FullAttention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new FullAttention(sourceAxis, targetAxis, params, ScaledDotProduct())

  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
      attentionScore: AttentionScore[Target, Source, Query, Key, V]
  ): FullAttention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new FullAttention(sourceAxis, targetAxis, params, attentionScore)

/** Attention where a target position may only attend to source positions up to its own index.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
class CausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool] = causalMask(scoreShape)

object CausalAttention:

  /** Defaults the attention scores to [[ScaledDotProduct]]. */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V]
  ): CausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new CausalAttention(sourceAxis, targetAxis, params, ScaledDotProduct())

  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
      attentionScore: AttentionScore[Target, Source, Query, Key, V]
  ): CausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new CausalAttention(sourceAxis, targetAxis, params, attentionScore)

/** Attention restricted by a caller-supplied mask.
  *
  * @param mask A function generating a boolean mask to prevent attention to certain positions.
  */
class CustomAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    mask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool] = mask(scoreShape)

object CustomAttention:

  /** Defaults the attention scores to [[ScaledDotProduct]]. */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
      mask: Shape2[Target, Source] => Tensor2[Target, Source, Bool]
  ): CustomAttention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new CustomAttention(params, mask, ScaledDotProduct())

  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
      params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
      mask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
      attentionScore: AttentionScore[Target, Source, Query, Key, V]
  ): CustomAttention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V] =
    new CustomAttention(params, mask, attentionScore)
