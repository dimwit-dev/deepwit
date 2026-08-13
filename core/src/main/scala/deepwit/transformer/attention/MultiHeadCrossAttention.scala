package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.softmax
import dimwit.stats.Normal
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.init
import dimwit.Label as Λ
import deepwit.transformer.fullMask

case class MultiHeadCrossAttention[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadCrossAttention.Params[CrossEmbedding, Embedding, V],
    createAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Bool]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val heads = headAttention(crossContext, context)
    heads.vmap(Axis[Context])(heads => headProjection(heads.flatten))

  protected def headAttention(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]) =
    zipvmap(Axis[Head])(params.wq, params.wk, params.wv):
      case (wq, wk, wv) =>
        val headAttention = CrossAttention(CrossAttention.Params(wq, wk, wv), createAttentionMask)
        headAttention(crossContext, context)

  protected def headProjection(headValues: Tensor1[Head |*| HeadValue, V]) = AffineLayer(params.headProjection)(headValues)

object MultiHeadCrossAttention:

  case class Params[CrossEmbedding, Embedding, V: IsFloating](
      wq: Tensor3[Head, Embedding, HeadQuery, V],
      wk: Tensor3[Head, CrossEmbedding, HeadKey, V],
      wv: Tensor3[Head, CrossEmbedding, HeadValue, V],
      headProjection: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[CrossEmbedding, Embedding, V] =
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val nHeads = headExtent.size
      val headProjectionGain = Math.sqrt(1.0 / (2 * numTransformerLayers)).toFloat
      Params(
        wq = stack(queryKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headQueryExtent, vtype, key)), Axis[Head]),
        wk = stack(keyKey.split(nHeads).map(key => init.xavierUniform(crossEmbeddingExtent, headKeyExtent, vtype, key)), Axis[Head]),
        wv = stack(valueKey.split(nHeads).map(key => init.xavierUniform(crossEmbeddingExtent, headValueExtent, vtype, key)), Axis[Head]),
        headProjection = AffineLayer.Params.xavierUniform(headExtent * headValueExtent, embeddingExtent, vtype, projectionKey, gain = headProjectionGain)
      )
