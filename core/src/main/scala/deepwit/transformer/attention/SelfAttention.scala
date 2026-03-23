package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.ActivationFunction.softmax
import deepwit.base.LinearLayer
import deepwit.transformer.causalMask

trait SelfAttention[Context: Label, Embedding: Label, Q: Label, K: Label, V: Label](
    params: SelfAttention.Params[Embedding, Q, K, V]
) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, V, Float]):

  protected def encodeToQuery(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wq)(embedding)
  protected def encodeToKey(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wk)(embedding)
  protected def encodeToValue(embedding: Tensor1[Embedding, Float]) = LinearLayer(params.wv)(embedding)

  protected def calculateAttentionScores(queries: Tensor2[Context, Q, Float], keys: Tensor2[Context, K, Float]): Tensor2[Context, Prime[Context], Float] =
    val dk = Math.sqrt(keys.shape(Axis[K])).toFloat
    queries.dot(Axis[Q ~ K])(keys) /! dk

  protected def calculateAttentionWeights(attentionScores: Tensor2[Context, Prime[Context], Float]): Tensor2[Context, AttentionWeights, Float]

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, V, Float] =
    val queries = context.vmap(Axis[Context])(encodeToQuery)
    val keys = context.vmap(Axis[Context])(encodeToKey)
    val values = context.vmap(Axis[Context])(encodeToValue)
    val attentionScores = calculateAttentionScores(queries, keys)
    val attentionWeights = calculateAttentionWeights(attentionScores)
    val res = attentionWeights.dot(Axis[AttentionWeights ~ Context])(values)
    res

case class FullSelfAttention[Context: Label, Embedding: Label, Q: Label, K: Label, V: Label](
    params: SelfAttention.Params[Embedding, Q, K, V]
) extends SelfAttention[Context, Embedding, Q, K, V](params):

  protected override def calculateAttentionWeights(attentionScores: Tensor2[Context, Prime[Context], Float]): Tensor2[Context, AttentionWeights, Float] =
    attentionScores.vmap(Axis[Context])(attentionScore =>
      softmax(attentionScore).relabelTo(Axis[AttentionWeights])
    )

case class CausalSelfAttention[Context: Label, Embedding: Label, Q: Label, K: Label, V: Label](
    params: SelfAttention.Params[Embedding, Q, K, V]
) extends SelfAttention[Context, Embedding, Q, K, V](params):

  protected override def calculateAttentionWeights(attentionScores: Tensor2[Context, Prime[Context], Float]): Tensor2[Context, AttentionWeights, Float] =
    val noScores = Tensor.like(attentionScores).fill(Float.NegativeInfinity)
    val maskedAttentionScores = where(causalMask(attentionScores.shape), attentionScores, noScores)
    maskedAttentionScores.vmap(Axis[Context])(attentionScore =>
      softmax(attentionScore).relabelTo(Axis[AttentionWeights])
    )

object SelfAttention:

  case class Params[Embedding, Q, K, V](
      wq: LinearLayer.Params[Embedding, Q],
      wk: LinearLayer.Params[Embedding, K],
      wv: LinearLayer.Params[Embedding, V]
  )

  object Params:
    def apply[E, Q, K, V](wq: Tensor2[E, Q, Float], wk: Tensor2[E, K, Float], wv: Tensor2[E, V, Float]): Params[E, Q, K, V] =
      Params(LinearLayer.Params(wq), LinearLayer.Params(wk), LinearLayer.Params(wv))

    def init[Embedding: Label, Q: Label, K: Label, V: Label](queryExtent: AxisExtent[Q], keyExtent: AxisExtent[K], valueExtent: AxisExtent[V], embeddingExtent: AxisExtent[Embedding], key: Random.Key): Params[Embedding, Q, K, V] =
      val (queryKey, keyKey, valueKey) = key.splitToTuple(3)
      Params(
        wq = LinearLayer.Params.xavierUniform(embeddingExtent, queryExtent, queryKey),
        wk = LinearLayer.Params.xavierUniform(embeddingExtent, keyExtent, keyKey),
        wv = LinearLayer.Params.xavierUniform(embeddingExtent, valueExtent, valueKey)
      )
