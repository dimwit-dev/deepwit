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
    val targetPositions = Tensor1(Axis[Target]).fromRange(0 until target.shape(Axis[Target]))
    val sourcePositions = Tensor1(Axis[Source]).fromRange(0 until source.shape(Axis[Source]))
    val attended = Attention.attend(attentionScore, createAttentionMask)(queries, keys, values, targetPositions, sourcePositions)
    (attended, (queries = queries, keys = keys, values = values))

  /** Creates a boolean mask indicating which source positions each target position may attend to.
    *
    * @return A boolean tensor where `mask(row, col) == true` means the target position
    *         at `row` is allowed to attend to the source position at `col`.
    */
  protected def createAttentionMask(targetPositions: Tensor1[Target, Int32], sourcePositions: Tensor1[Source, Int32]): Tensor2[Target, Source, Bool]

object Attention:

  /** Attends from the `queries` to the `keys` and `values`: scores them with `attentionScore`, hides what
    * `attentionMask` does not allow, and averages the values weighted by the softmax of the scores.
    *
    * @param targetPositions The positions of the queries, e.g. a single one when decoding with a [[KVCache]].
    * @param sourcePositions The positions of the keys and values.
    */
  def attend[Source: Λ, Target: Λ, Query, Key, Value: Λ, V: IsFloating](
      attentionScore: AttentionScore[Target, Source, Query, Key, V],
      attentionMask: AttentionMask[Target, Source]
  )(
      queries: Tensor2[Target, Query, V],
      keys: Tensor2[Source, Key, V],
      values: Tensor2[Source, Value, V],
      targetPositions: Tensor1[Target, Int32],
      sourcePositions: Tensor1[Source, Int32]
  ): Tensor2[Target, Value, V] =
    val attentionScores = attentionScore(queries, keys)
    val maskedScores = where_!(attentionMask(targetPositions, sourcePositions), attentionScores, Float.NegativeInfinity)
    val attentionWeights = maskedScores.vapply(Axis[Source])(softmax)
    attentionWeights.dot(Axis[Source])(values)

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

  override protected def createAttentionMask(targetPositions: Tensor1[Target, Int32], sourcePositions: Tensor1[Source, Int32]): Tensor2[Target, Source, Bool] =
    fullMask(targetPositions, sourcePositions)

/** Attention where a target position may only attend to source positions up to its own index. */
class CausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(targetPositions: Tensor1[Target, Int32], sourcePositions: Tensor1[Source, Int32]): Tensor2[Target, Source, Bool] =
    causalMask(targetPositions, sourcePositions)

/** @param attentionMask Creates a boolean mask indicating which source positions each target position may attend to. */
class CustomAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: Attention.Params[SourceEmbedding, TargetEmbedding, Query, Key, Value, V],
    attentionMask: AttentionMask[Target, Source],
    attentionScore: AttentionScore[Target, Source, Query, Key, V]
) extends Attention[Source, SourceEmbedding, Target, TargetEmbedding, Query, Key, Value, V](params, attentionScore):

  override protected def createAttentionMask(targetPositions: Tensor1[Target, Int32], sourcePositions: Tensor1[Source, Int32]): Tensor2[Target, Source, Bool] =
    attentionMask(targetPositions, sourcePositions)

/** A target position may attend to the source positions up to and including its own. */
private def causalMask[Target: Λ, Source: Λ](targetPositions: Tensor1[Target, Int32], sourcePositions: Tensor1[Source, Int32]): Tensor2[Target, Source, Bool] =
  targetPositions.vmap(Axis[Target])(targetPosition => sourcePositions <=! targetPosition)

/** A target position may attend to every source position. */
private def fullMask[Target: Λ, Source: Λ](targetPositions: Tensor1[Target, Int32], sourcePositions: Tensor1[Source, Int32]): Tensor2[Target, Source, Bool] =
  Tensor(Shape(targetPositions.shape.extent(Axis[Target]), sourcePositions.shape.extent(Axis[Source]))).fill(true)
