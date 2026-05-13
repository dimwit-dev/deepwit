package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue, MultiHeadSelfAttention, MultiHeadCausalSelfAttention, MultiHeadCrossAttention}

class CrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label, V: IsFloating](
    hyperParams: CrossTransformerLayer.HyperParams[CrossContext, Context, Embedding]
)(
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding, V]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  private val selfAttention = MultiHeadCausalSelfAttention[Context, Embedding, V](params.selfAttentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNormParams)

  private val crossAttention = new MultiHeadCrossAttention(hyperParams.multiHeadCrossAttention)(params.crossAttentionParams)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    var x = context
    x = x + contextMixer(x)
    x = x + crossContextMixer(crossContext, x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

  protected def embeddingMixer(embeddings: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embeddings))

  protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

  protected def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    crossAttention(crossContext, context.vmap(Axis[Context])(crossAttentionPreNorm))

object CrossTransformerLayer:

  def apply[CrossContext: Label, Context: Label, CrossEmbedding: Label, Embedding: Label, V: IsFloating](hyperParams: HyperParams[CrossContext, Context, Embedding])(params: Params[CrossEmbedding, Embedding, V]): CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding, V] =
    new CrossTransformerLayer(hyperParams)(params)

  case class HyperParams[CrossContext: Label, Context: Label, Embedding: Label](
      multiHeadCrossAttention: MultiHeadCrossAttention.HyperParams[CrossContext, Context]
  )

  case class Params[CrossEmbedding, Embedding, V: IsFloating](
      crossAttentionParams: MultiHeadCrossAttention.Params[CrossEmbedding, Embedding, V],
      crossAttentionNormParams: LayerNorm.Params[Embedding, V],
      selfAttentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNormParams: LayerNorm.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Label, Embedding: Label, V: IsFloating](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[CrossEmbedding, Embedding, V] =
      val (selfAttnKey, crossAttnKey, mixKey) = key.splitToTuple(3)
      new Params[CrossEmbedding, Embedding, V](
        crossAttentionParams = MultiHeadCrossAttention.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, crossEmbeddingExtent, embeddingExtent, vtype, crossAttnKey),
        crossAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        selfAttentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, vtype, selfAttnKey),
        selfAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
