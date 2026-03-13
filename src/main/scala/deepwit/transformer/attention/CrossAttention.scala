package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.ActivationFunction.softmax
import deepwit.base.LinearLayer

trait ICrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label, Q: Label, K: Label, V: Label] extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, V, Float]):

  def encodeToQuery(embedding: Tensor1[Embedding, Float]): Tensor1[Q, Float]
  def encodeToKey(embedding: Tensor1[CrossEmbedding, Float]): Tensor1[K, Float]
  def encodeToValue(embedding: Tensor1[CrossEmbedding, Float]): Tensor1[V, Float]
  def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[CrossContext, K, Float]): Tensor2[Context, AttentionWeights, Float]

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, V, Float] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = crossContext.vmap(Axis[CrossContext])(encodeToKey)
    val values = crossContext.vmap(Axis[CrossContext])(encodeToValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    val res = attentionWeights.dot(Axis[AttentionWeights ~ CrossContext])(values)
    res

case class CrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label, Q: Label, K: Label, V: Label](
    hyperParams: CrossAttention.HyperParams[CrossContext, Context]
)(
    params: CrossAttention.BaseParams[CrossEmbedding, Embedding, Q, K, V]
) extends ICrossAttention[CrossContext, CrossEmbedding, Context, Embedding, Q, K, V]:

  override def encodeToQuery(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wq)(embedding)
  override def encodeToKey(embedding: Tensor1[CrossEmbedding, Float]) = LinearLayer(params.wk)(embedding)
  override def encodeToValue(embedding: Tensor1[CrossEmbedding, Float]) = LinearLayer(params.wv)(embedding)

  def calculateAttentionScores(queries: Tensor2[Context, Q, Float], keys: Tensor2[CrossContext, K, Float]): Tensor2[Context, CrossContext, Float] =
    val dk = Math.sqrt(keys.shape(Axis[K])).toFloat
    queries.dot(Axis[Q ~ K])(keys) /! dk

  override def calculateAttentionWeights(queries: Tensor2[Context, Q, Float], keys: Tensor2[CrossContext, K, Float]) =
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = where(hyperParams.createAttentionMask(attentionScores.shape), attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
      .vmap(Axis[Context])(attentionScore => softmax(attentionScore).relabelTo(Axis[AttentionWeights]))
    attentionWeights

object CrossAttention:

  case class HyperParams[CrossContext, Context](
      createAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Boolean]
  )

  case class BaseParams[CrossEmbedding, Embedding, Q, K, V](
      wq: LinearLayer.Params[Embedding, Q],
      wk: LinearLayer.Params[CrossEmbedding, K],
      wv: LinearLayer.Params[CrossEmbedding, V]
  )

  object BaseParams:

    def apply[CE, E, Q, K, V](wq: Tensor2[E, Q, Float], wk: Tensor2[CE, K, Float], wv: Tensor2[CE, V, Float]): BaseParams[CE, E, Q, K, V] =
      new BaseParams(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[CrossEmbedding: Label, Embedding: Label, Q: Label, K: Label, V: Label](queryExtent: AxisExtent[Q], keyExtent: AxisExtent[K], valueExtent: AxisExtent[V], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], key: Random.Key): BaseParams[CrossEmbedding, Embedding, Q, K, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      BaseParams(
        wq = LinearLayer.Params.xavierUniform(embeddingExtent, queryExtent, queryKey),
        wk = LinearLayer.Params.xavierUniform(crossEmbeddingExtent, keyExtent, keyKey),
        wv = LinearLayer.Params.xavierUniform(crossEmbeddingExtent, valueExtent, valueKey)
      )
