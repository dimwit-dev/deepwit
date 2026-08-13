package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.softmax
import deepwit.base.LinearLayer
import dimwit.Label as Λ

case class CrossAttention[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: CrossAttention.Params[CrossEmbedding, Embedding, Query, Key, Value, V],
    createAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Bool]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Value, V]):

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Value, V] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = crossContext.vmap(Axis[CrossContext])(encodeToKey)
    val values = crossContext.vmap(Axis[CrossContext])(encodeToValue)
    val attentionWeights = calculateAttentionWeights(queries, keys)
    val res = attentionWeights.dot(Axis[AttentionWeights] -> Axis[CrossContext])(values)
    res

  protected def encodeToQuery(embedding: Tensor1[Embedding, V]) = LinearLayer(params.wq)(embedding)
  protected def encodeToKey(embedding: Tensor1[CrossEmbedding, V]) = LinearLayer(params.wk)(embedding)
  protected def encodeToValue(embedding: Tensor1[CrossEmbedding, V]) = LinearLayer(params.wv)(embedding)

  protected def calculateAttentionScores(queries: Tensor2[Context, Query, V], keys: Tensor2[CrossContext, Key, V]): Tensor2[Context, CrossContext, V] =
    val dk = Math.sqrt(keys.shape(Axis[Key])).toFloat
    queries.dot(Axis[Query] -> Axis[Key])(keys) /! dk

  protected def calculateAttentionWeights(queries: Tensor2[Context, Query, V], keys: Tensor2[CrossContext, Key, V]) =
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = where(createAttentionMask(attentionScores.shape), attentionScores, Tensor.like(attentionScores).fill(Float.NegativeInfinity))
      .vmap(Axis[Context])(attentionScore => softmax(attentionScore).relabelTo(Axis[AttentionWeights]))
    attentionWeights

object CrossAttention:

  case class Params[CrossEmbedding, Embedding, Query, Key, Value, V: IsFloating](
      wq: LinearLayer.Params[Embedding, Query, V],
      wk: LinearLayer.Params[CrossEmbedding, Key, V],
      wv: LinearLayer.Params[CrossEmbedding, Value, V]
  )

  object Params:

    def apply[CrossEmbedding, Embedding, Query, Key, Value, V: IsFloating](wq: Tensor2[Embedding, Query, V], wk: Tensor2[CrossEmbedding, Key, V], wv: Tensor2[CrossEmbedding, Value, V]): Params[CrossEmbedding, Embedding, Query, Key, Value, V] =
      new Params(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[CrossEmbedding: Λ, Embedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](queryExtent: AxisExtent[Query], keyExtent: AxisExtent[Key], valueExtent: AxisExtent[Value], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[CrossEmbedding, Embedding, Query, Key, Value, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      Params(
        wq = LinearLayer.Params.xavierUniform(embeddingExtent, queryExtent, vtype, queryKey),
        wk = LinearLayer.Params.xavierUniform(crossEmbeddingExtent, keyExtent, vtype, keyKey),
        wv = LinearLayer.Params.xavierUniform(crossEmbeddingExtent, valueExtent, vtype, valueKey)
      )
