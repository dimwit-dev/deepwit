package deepwit.attention

import dimwit.*
import deepwit.activation.softmax
import deepwit.base.LinearLayer
import dimwit.Label as Λ

/** Represents an attention mechanism from a target sequence of embeddings onto a source sequence of embeddings.
  *
  * The queries are projected from the target sequence, the keys and values from the source sequence.
  * Which source positions a target position may attend to is left to the implementation: * [[FullAttention]], [[CausalAttention]] or [[CustomAttention]].
  */
abstract class Attention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, TargetEmbedding, V]) => Tensor2[Target, Value, V]):

  private val projectQuery = LinearLayer(params.queryWeights)
  private val projectKey = LinearLayer(params.keyWeights)
  private val projectValue = LinearLayer(params.valueWeights)

  final def apply(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): Tensor2[Target, Value, V] =
    applyWithIntermediates(source, target).head

  /** Computes the attention output alongside intermediate projections.
    *
    * This method performs the same core computation as [[apply]], but exposes
    * the intermediate query, key, and value tensors for further inspection.
    */
  final def applyWithIntermediates(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): (Tensor2[Target, Value, V], Attention.Intermediates[Source, Target, Query, Key, Value, V]) =
    val queries = target.vmap(Axis[Target])(projectQuery)
    val keys = source.vmap(Axis[Source])(projectKey)
    val values = source.vmap(Axis[Source])(projectValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    val attentionValues = attentionWeights.dot(Axis[Source])(values)
    (attentionValues, (queries = queries, keys = keys, values = values))

  private final def calculateAttentionWeights(queries: Tensor2[Target, Query, V], keys: Tensor2[Source, Key, V]) =
    val attentionScores = attentionScore(queries, keys)
    val attentionMask = createAttentionMask(attentionScores.shape)
    val maskedScores = where(attentionMask, attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
    maskedScores.vapply(Axis[Source])(softmax)

  /** Creates a boolean mask indicating which source positions each target position may attend to.
    *
    * @param scoreShape The shape of the attention scores (same shape as attention weights).
    * @return A boolean tensor where `mask(row, col) == true` means the target position
    *         at `row` is allowed to attend to the source position at `col`.
    */
  protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool]

object Attention:

  /** Holds the intermediate tensors computed during attention. */
  type Intermediates[Source, Target, Query, Key, Value, V] = (
      queries: Tensor2[Target, Query, V],
      keys: Tensor2[Source, Key, V],
      values: Tensor2[Source, Value, V]
  )

  /** Holds the parameters for the attention mechanism. */
  case class Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V](
      queryWeights: LinearLayer.Params[TargetEmbedding, Query, V],
      keyWeights: LinearLayer.Params[SourceEmbedding, Key, V],
      valueWeights: LinearLayer.Params[SourceEmbedding, Value, V]
  )

  object Params:

    /** Creates attention parameters from the given weight tensors. */
    def apply[SourceEmbedding, TargetEmbedding, Query, Key, Value, V: IsFloating](wq: Tensor2[TargetEmbedding, Query, V], wk: Tensor2[SourceEmbedding, Key, V], wv: Tensor2[SourceEmbedding, Value, V]): Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V] =
      new Params(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[SourceEmbedding: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](queryExtent: AxisExtent[Query], keyExtent: AxisExtent[Key], valueExtent: AxisExtent[Value], sourceEmbeddingExtent: AxisExtent[SourceEmbedding], targetEmbeddingExtent: AxisExtent[TargetEmbedding], key: Random.Key, vtype: VType[V] = VType[Float32]): Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V] =
      xavierUniform(queryExtent, keyExtent, valueExtent, sourceEmbeddingExtent, targetEmbeddingExtent, key, vtype)

    def xavierUniform[SourceEmbedding: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](queryExtent: AxisExtent[Query], keyExtent: AxisExtent[Key], valueExtent: AxisExtent[Value], sourceEmbeddingExtent: AxisExtent[SourceEmbedding], targetEmbeddingExtent: AxisExtent[TargetEmbedding], key: Random.Key, vtype: VType[V] = VType[Float32]): Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      Params(
        queryWeights = LinearLayer.Params.xavierUniform(targetEmbeddingExtent, queryExtent, queryKey, vtype),
        keyWeights = LinearLayer.Params.xavierUniform(sourceEmbeddingExtent, keyExtent, keyKey, vtype),
        valueWeights = LinearLayer.Params.xavierUniform(sourceEmbeddingExtent, valueExtent, valueKey, vtype)
      )

/** Attention where every target position may attend to every source position. */
class FullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool] = fullMask(scoreShape)

/** Attention where a target position may only attend to source positions up to its own index. */
class CausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool] = causalMask(scoreShape)

/** @param mask Creates a boolean mask indicating which source positions each target position may attend to. */
class CustomAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(scoreShape: Shape2[Target, Source]): Tensor2[Target, Source, Bool] = attentionMask(scoreShape)

private def causalMask[Context: Λ, CrossContext: Λ](scoreShape: Shape2[Context, CrossContext]): Tensor[(Context, CrossContext), Bool] = tril(fullMask(scoreShape))
private def fullMask[Context: Λ, CrossContext: Λ](scoreShape: Shape2[Context, CrossContext]): Tensor[(Context, CrossContext), Bool] = Tensor(scoreShape).fill(true)
