package deepwit.transformer

import dimwit.*
import deepwit.base.ActivationFunction.gelu
import deepwit.normalization.LayerNorm
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue, MultiHeadSelfAttention, MultiHeadCausalSelfAttention, MultiHeadFullSelfAttention}

trait TransformerLayer[Context: Label, Embedding: Label, V: IsFloating](
    params: TransformerLayer.Params[Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  def selfAttention: MultiHeadSelfAttention[Context, Embedding, V] // = MultiHeadSelfAttention(hyperParams.multiHeadAttention)(params.attentionParams)

  val selfAttentionNorm = LayerNorm(params.attentionNormParams)

  val mlp = MLPEmbeddingMixer(params.mlpParams)
  val mlpNorm = LayerNorm(params.mlpNormParams)

  private def embeddingMixer(embeddings: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    val embNorm = mlpNorm(embeddings)
    mlp(embNorm)

  private def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val contextNorm = context.vmap(Axis[Context])(selfAttentionNorm)
    selfAttention(contextNorm)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    var x = context
    x = x + contextMixer(x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

case class CausalTransformerLayer[Context: Label, Embedding: Label, V: IsFloating](
    params: TransformerLayer.Params[Embedding, V]
) extends TransformerLayer[Context, Embedding, V](params):
  override val selfAttention = MultiHeadCausalSelfAttention[Context, Embedding, V](params.attentionParams)

case class BidirectionalTransformerLayer[Context: Label, Embedding: Label, V: IsFloating](
    params: TransformerLayer.Params[Embedding, V]
) extends TransformerLayer[Context, Embedding, V](params):
  override val selfAttention = MultiHeadFullSelfAttention[Context, Embedding, V](params.attentionParams)

object TransformerLayer:

  case class Params[Embedding, V](
      attentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      attentionNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Label, V: IsFloating](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (attnKey, mixKey) = key.splitToTuple(2)
      new Params[Embedding, V](
        attentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, vtype, attnKey),
        attentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
