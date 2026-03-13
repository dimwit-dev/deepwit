package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.ActivationFunction.softmax
import dimwit.stats.Normal
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.init

trait IMultiHeadCrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label] extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):

  def headAttention(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor[(Head, Context, HeadValue), Float]
  def headProjection(headValues: Tensor1[Head |*| HeadValue, Float]): Tensor1[Embedding, Float]

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val heads = headAttention(crossContext, context)
    heads.vmap(Axis[Context])(heads => headProjection(heads.flatten))

case class MultiHeadCrossAttention[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
    hyperParams: MultiHeadCrossAttention.HyperParams[CrossContext, Context]
)(
    params: MultiHeadCrossAttention.Params[CrossEmbedding, Embedding]
) extends IMultiHeadCrossAttention[CrossContext, CrossEmbedding, Context, Embedding]:

  override def headAttention(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]) =
    zipvmap(Axis[Head])(params.wq, params.wk, params.wv):
      case (wq, wk, wv) =>
        val headAttention = CrossAttention(hyperParams.headAttention)(CrossAttention.BaseParams(wq, wk, wv))
        headAttention(crossContext, context)

  override def headProjection(headValues: Tensor1[Head |*| HeadValue, Float]) = AffineLayer(params.headProjection)(headValues)

object MultiHeadCrossAttention:

  case class HyperParams[CrossContext, Context](
      val headAttention: CrossAttention.HyperParams[CrossContext, Context]
  )

  case class Params[CrossEmbedding, Embedding](
      wq: Tensor3[Head, Embedding, HeadQuery, Float],
      wk: Tensor3[Head, CrossEmbedding, HeadKey, Float],
      wv: Tensor3[Head, CrossEmbedding, HeadValue, Float],
      headProjection: AffineLayer.Params[Head |*| HeadValue, Embedding]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Label, Embedding: Label](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], key: Random.Key): Params[CrossEmbedding, Embedding] =
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val nHeads = headExtent.size
      val headProjectionGain = Math.sqrt(1.0 / (2 * numTransformerLayers)).toFloat
      Params(
        wq = stack(queryKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headQueryExtent, key)), Axis[Head]),
        wk = stack(keyKey.split(nHeads).map(key => init.xavierUniform(crossEmbeddingExtent, headKeyExtent, key)), Axis[Head]),
        wv = stack(valueKey.split(nHeads).map(key => init.xavierUniform(crossEmbeddingExtent, headValueExtent, key)), Axis[Head]),
        headProjection = AffineLayer.Params.xavierUniform(headExtent * headValueExtent, embeddingExtent, projectionKey, gain = headProjectionGain)
      )
