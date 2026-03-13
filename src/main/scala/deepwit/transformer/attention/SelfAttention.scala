package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.ActivationFunction.softmax
import deepwit.base.LinearLayer

trait ISelfAttention[Context: Label, Embedding: Label, Q: Label, K: Label, V: Label] extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, V, Float]):

  def encodeToQuery(embedding: Tensor1[Embedding, Float]): Tensor1[Q, Float]
  def encodeToKey(embedding: Tensor1[Embedding, Float]): Tensor1[K, Float]
  def encodeToValue(embedding: Tensor1[Embedding, Float]): Tensor1[V, Float]
  def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[Context, K, Float]): Tensor2[Context, AttentionWeights, Float]

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, V, Float] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = context.vmap(Axis[Context])(encodeToKey)
    val values = context.vmap(Axis[Context])(encodeToValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    val res = attentionWeights.dot(Axis[AttentionWeights ~ Context])(values)
    res

case class SelfAttention[Context: Label, Embedding: Label, Q: Label, K: Label, V: Label](
    hyperParams: SelfAttention.HyperParams[Context]
)(
    params: SelfAttention.BaseParams[Embedding, Q, K, V]
) extends ISelfAttention[Context, Embedding, Q, K, V]:

  override def encodeToQuery(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wq)(embedding)
  override def encodeToKey(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wk)(embedding)
  override def encodeToValue(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wv)(embedding)

  def calculateAttentionScores(queries: Tensor2[Context, Q, Float], keys: Tensor2[Context, K, Float]): Tensor2[Context, Prime[Context], Float] =
    val dk = Math.sqrt(keys.shape(Axis[K])).toFloat
    queries.dot(Axis[Q ~ K])(keys) /! dk

  override def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[Context, K, Float]) =
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = where(hyperParams.createAttentionMask(attentionScores.shape), attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
      .vmap(Axis[Context])(attentionScore => softmax(attentionScore).relabelTo(Axis[AttentionWeights]))
    attentionWeights

object SelfAttention:

  case class HyperParams[Context](
      createAttentionMask: Shape2[Context, Prime[Context]] => Tensor[(Context, Prime[Context]), Boolean]
      // calculateAttentionWeights: (Tensor2[Context, Q, Float], Tensor2[Context, K, Float]) => Tensor2[Context, AttentionWeights, Float]
  )

  case class BaseParams[Embedding, Q, K, V](
      wq: LinearLayer.Params[Embedding, Q],
      wk: LinearLayer.Params[Embedding, K],
      wv: LinearLayer.Params[Embedding, V]
  )

  object BaseParams:
    def apply[E, Q, K, V](wq: Tensor2[E, Q, Float], wk: Tensor2[E, K, Float], wv: Tensor2[E, V, Float]): BaseParams[E, Q, K, V] =
      BaseParams(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[Embedding: Label, Q: Label, K: Label, V: Label](queryExtent: AxisExtent[Q], keyExtent: AxisExtent[K], valueExtent: AxisExtent[V], embeddingExtent: AxisExtent[Embedding], key: Random.Key): BaseParams[Embedding, Q, K, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      BaseParams(
        wq = LinearLayer.Params.xavierUniform(embeddingExtent, queryExtent, queryKey),
        wk = LinearLayer.Params.xavierUniform(embeddingExtent, keyExtent, keyKey),
        wv = LinearLayer.Params.xavierUniform(embeddingExtent, valueExtent, valueKey)
      )
