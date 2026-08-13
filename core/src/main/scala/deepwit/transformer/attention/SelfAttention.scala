package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.softmax
import deepwit.base.LinearLayer
import deepwit.transformer.{fullMask, causalMask}
import dimwit.Label as Λ

class SelfAttention[Context: Λ, Embedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](
    params: SelfAttention.Params[Embedding, Query, Key, Value, V],
    createAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Value, V]):

  protected def encodeToQuery(embedding: Tensor1[Embedding, V]) = LinearLayer(params.wq)(embedding)
  protected def encodeToKey(embedding: Tensor1[Embedding, V]) = LinearLayer(params.wk)(embedding)
  protected def encodeToValue(embedding: Tensor1[Embedding, V]) = LinearLayer(params.wv)(embedding)

  protected def calculateAttentionScores(queries: Tensor2[Context, Query, V], keys: Tensor2[Context, Key, V]): Tensor2[Context, Prime[Context], V] =
    val dk = Math.sqrt(keys.shape(Axis[Key])).toFloat
    queries.dot(Axis[Query] -> Axis[Key])(keys) /! dk

  protected def calculateAttentionWeights(attentionScores: Tensor2[Context, Prime[Context], V]): Tensor2[Context, AttentionWeights, V] =
    val noScores = Tensor.like(attentionScores).fill(Float.NegativeInfinity)
    val maskedAttentionScores = where(causalMask(attentionScores.shape), attentionScores, noScores)
    maskedAttentionScores.vmap(Axis[Context])(attentionScore =>
      softmax(attentionScore).relabelTo(Axis[AttentionWeights])
    )

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Value, V] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = context.vmap(Axis[Context])(encodeToKey)
    val values = context.vmap(Axis[Context])(encodeToValue)
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = calculateAttentionWeights(attentionScores)
    val res = attentionWeights.dot(Axis[AttentionWeights] -> Axis[Context])(values)
    res

object SelfAttention:

  case class Params[Embedding, Query, Key, Value, V: IsFloating](
      wq: LinearLayer.Params[Embedding, Query, V],
      wk: LinearLayer.Params[Embedding, Key, V],
      wv: LinearLayer.Params[Embedding, Value, V]
  )

  object Params:
    def apply[Embedding, Query, Key, Value, V: IsFloating](wq: Tensor2[Embedding, Query, V], wk: Tensor2[Embedding, Key, V], wv: Tensor2[Embedding, Value, V]): Params[Embedding, Query, Key, Value, V] =
      Params(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[Embedding: Λ, Query: Λ, Key: Λ, Value: Λ, V: IsFloating](queryExtent: AxisExtent[Query], keyExtent: AxisExtent[Key], valueExtent: AxisExtent[Value], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[Embedding, Query, Key, Value, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      Params(
        wq = LinearLayer.Params.xavierUniform(embeddingExtent, queryExtent, vtype, queryKey),
        wk = LinearLayer.Params.xavierUniform(embeddingExtent, keyExtent, vtype, keyKey),
        wv = LinearLayer.Params.xavierUniform(embeddingExtent, valueExtent, vtype, valueKey)
      )
