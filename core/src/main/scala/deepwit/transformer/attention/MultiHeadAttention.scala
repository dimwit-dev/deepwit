package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.softmax
import dimwit.stats.Normal
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.init
import dimwit.Label as Λ
import deepwit.transformer.fullMask

case class MultiHeadAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadAttention.Params[SourceEmbedding, Embedding, V],
    createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, Embedding, V]) => Tensor2[Target, Embedding, V]):

  private val projectHeadValues = AffineLayer(params.headWeights)

  override def apply(crossContext: Tensor2[Source, SourceEmbedding, V], context: Tensor2[Target, Embedding, V]): Tensor2[Target, Embedding, V] =
    val heads = headAttention(crossContext, context)
    heads.vmap(Axis[Target])(heads => projectHeadValues(heads.flatten))

  protected def headAttention(crossContext: Tensor2[Source, SourceEmbedding, V], context: Tensor2[Target, Embedding, V]) =
    zipvmap(Axis[Head])(params.queryWeights, params.keyWeights, params.valueWeights):
      case (q, k, v) =>
        val headAttention = Attention(Attention.Params(q, k, v), createAttentionMask)
        headAttention(crossContext, context)

object MultiHeadAttention:

  case class Params[SourceEmbedding, Embedding, V](
      queryWeights: Tensor3[Head, Embedding, HeadQuery, V], // TODO update to List[Attention.Params] => Check performance
      keyWeights: Tensor3[Head, SourceEmbedding, HeadKey, V],
      valueWeights: Tensor3[Head, SourceEmbedding, HeadValue, V],
      headWeights: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  )

  object Params:

    def xavierUniformHeads[Embedding: Λ, HeadOut: Λ, V: IsFloating](numHeads: Int, embeddingExtent: AxisExtent[Embedding], headExtent: AxisExtent[HeadOut], vtype: VType[V], key: Random.Key): Tensor3[Head, Embedding, HeadOut, V] =
      stack(key.split(numHeads).map(key => init.xavierUniform(embeddingExtent, headExtent, vtype, key)), Axis[Head])

    def xavierUniformHeadWeights[In: Λ, Out: Λ, V: IsFloating](numLayers: Int, embeddingExtent: AxisExtent[In], headExtent: AxisExtent[Out], vtype: VType[V], key: Random.Key): AffineLayer.Params[In, Out, V] =
      val gain = Math.sqrt(1.0 / (2 * numLayers)).toFloat
      AffineLayer.Params.xavierUniform(embeddingExtent, headExtent, vtype, key, gain = gain)

    def xavierUniformDepthScaled[SourceEmbedding: Λ, Embedding: Λ, V: IsFloating](numTransformerLayers: Int, headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[SourceEmbedding], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[SourceEmbedding, Embedding, V] =
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headQueryExtent, vtype, queryKey),
        keyWeights = xavierUniformHeads(headExtent.size, crossEmbeddingExtent, headKeyExtent, vtype, keyKey),
        valueWeights = xavierUniformHeads(headExtent.size, crossEmbeddingExtent, headValueExtent, vtype, valueKey),
        headWeights = xavierUniformHeadWeights(numTransformerLayers, headExtent * headValueExtent, embeddingExtent, vtype, projectionKey)
      )
