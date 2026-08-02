package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.ActivationFunction.softmax
import deepwit.base.LinearLayer
import deepwit.transformer.causalMask

trait SelfAttention[Context: Label, Embedding: Label, Query: Label, Key: Label, Value: Label, V: IsFloating](
    params: SelfAttention.Params[Embedding, Query, Key, Value, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Value, V]):

  protected def encodeToQuery(embedding: Tensor1[Embedding, V]) = LinearLayer(params.wq)(embedding)
  protected def encodeToKey(embedding: Tensor1[Embedding, V]) = LinearLayer(params.wk)(embedding)
  protected def encodeToValue(embedding: Tensor1[Embedding, V]) = LinearLayer(params.wv)(embedding)

  protected def calculateAttentionScores(queries: Tensor2[Context, Query, V], keys: Tensor2[Context, Key, V]): Tensor2[Context, Prime[Context], V] =
    val dk = Math.sqrt(keys.shape(Axis[Key])).toFloat
    queries.dot(Axis[Query] -> Axis[Key])(keys) /! dk

  protected def calculateAttentionWeights(attentionScores: Tensor2[Context, Prime[Context], V]): Tensor2[Context, AttentionWeights, V]

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Value, V] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = context.vmap(Axis[Context])(encodeToKey)
    val values = context.vmap(Axis[Context])(encodeToValue)
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = calculateAttentionWeights(attentionScores)
    val res = attentionWeights.dot(Axis[AttentionWeights] -> Axis[Context])(values)
    res

case class FullSelfAttention[Context: Label, Embedding: Label, Query: Label, Key: Label, Value: Label, V: IsFloating](
    params: SelfAttention.Params[Embedding, Query, Key, Value, V]
) extends SelfAttention[Context, Embedding, Query, Key, Value, V](params):

  protected override def calculateAttentionWeights(attentionScores: Tensor2[Context, Prime[Context], V]): Tensor2[Context, AttentionWeights, V] =
    attentionScores.vmap(Axis[Context])(attentionScore =>
      softmax(attentionScore).relabelTo(Axis[AttentionWeights])
    )

case class CausalSelfAttention[Context: Label, Embedding: Label, Query: Label, Key: Label, Value: Label, V: IsFloating](
    params: SelfAttention.Params[Embedding, Query, Key, Value, V]
) extends SelfAttention[Context, Embedding, Query, Key, Value, V](params):

  protected override def calculateAttentionWeights(attentionScores: Tensor2[Context, Prime[Context], V]): Tensor2[Context, AttentionWeights, V] =
    val noScores = Tensor.like(attentionScores).fill(Float.NegativeInfinity)
    val maskedAttentionScores = where(causalMask(attentionScores.shape), attentionScores, noScores)
    maskedAttentionScores.vmap(Axis[Context])(attentionScore =>
      softmax(attentionScore).relabelTo(Axis[AttentionWeights])
    )

object SelfAttention:

  case class Params[Embedding, Query, Key, Value, V: IsFloating](
      wq: LinearLayer.Params[Embedding, Query, V],
      wk: LinearLayer.Params[Embedding, Key, V],
      wv: LinearLayer.Params[Embedding, Value, V]
  )

  object Params:
    def apply[Embedding, Query, Key, Value, V: IsFloating](wq: Tensor2[Embedding, Query, V], wk: Tensor2[Embedding, Key, V], wv: Tensor2[Embedding, Value, V]): Params[Embedding, Query, Key, Value, V] =
      Params(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[Embedding: Label, Query: Label, Key: Label, Value: Label, V: IsFloating](queryExtent: AxisExtent[Query], keyExtent: AxisExtent[Key], valueExtent: AxisExtent[Value], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[Embedding, Query, Key, Value, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      Params(
        wq = LinearLayer.Params.xavierUniform(embeddingExtent, queryExtent, vtype, queryKey),
        wk = LinearLayer.Params.xavierUniform(embeddingExtent, keyExtent, vtype, keyKey),
        wv = LinearLayer.Params.xavierUniform(embeddingExtent, valueExtent, vtype, valueKey)
      )
