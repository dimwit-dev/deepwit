package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.ActivationFunction.softmax
import dimwit.stats.Normal
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.init

trait MultiHeadSelfAttention[Context: Label, Embedding: Label](
    params: MultiHeadSelfAttention.Params[Embedding]
) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  private val headProjectionLayer = AffineLayer(params.headProjection)

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val heads = headAttention(context)
    heads.vmap(Axis[Context])(heads => headProjection(heads.flatten))

  protected def headAttention(context: Tensor2[Context, Embedding, Float]): Tensor[(Head, Context, HeadValue), Float] =
    zipvmap(Axis[Head])(params.wq, params.wk, params.wv):
      case (wq, wk, wv) =>
        selfAttention(SelfAttention.Params(wq, wk, wv))(context)

  protected def headProjection(headValues: Tensor1[Head |*| HeadValue, Float]) = headProjectionLayer(headValues)

  protected def selfAttention(params: SelfAttention.Params[Embedding, HeadQuery, HeadKey, HeadValue]): SelfAttention[Context, Embedding, HeadQuery, HeadKey, HeadValue]

case class MultiHeadCausalSelfAttention[Context: Label, Embedding: Label](
    params: MultiHeadSelfAttention.Params[Embedding]
) extends MultiHeadSelfAttention[Context, Embedding](params):

  protected override def selfAttention(params: SelfAttention.Params[Embedding, HeadQuery, HeadKey, HeadValue]) = CausalSelfAttention(params)

case class MultiHeadFullSelfAttention[Context: Label, Embedding: Label](
    params: MultiHeadSelfAttention.Params[Embedding]
) extends MultiHeadSelfAttention[Context, Embedding](params):

  protected override def selfAttention(params: SelfAttention.Params[Embedding, HeadQuery, HeadKey, HeadValue]) = FullSelfAttention(params)

object MultiHeadSelfAttention:

  case class Params[Embedding](
      wq: Tensor3[Head, Embedding, HeadQuery, Float],
      wk: Tensor3[Head, Embedding, HeadKey, Float],
      wv: Tensor3[Head, Embedding, HeadValue, Float],
      headProjection: AffineLayer.Params[Head |*| HeadValue, Embedding]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Label](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding], key: Random.Key): Params[Embedding] =
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val nHeads = headExtent.size
      val headProjectionGain = Math.sqrt(1.0 / (2 * numTransformerLayers)).toFloat
      Params(
        wq = stack(queryKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headQueryExtent, key)), Axis[Head]),
        wk = stack(keyKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headKeyExtent, key)), Axis[Head]),
        wv = stack(valueKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headValueExtent, key)), Axis[Head]),
        headProjection = AffineLayer.Params.xavierUniform(headExtent * headValueExtent, embeddingExtent, projectionKey, gain = headProjectionGain)
      )
