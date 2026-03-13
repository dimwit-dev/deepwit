package deepwit.transformer

import dimwit.*
import deepwit.base.ActivationFunction.gelu
import deepwit.normalization.LayerNorm
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue, MultiHeadSelfAttention, MultiHeadCrossAttention}

trait ICrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label] extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):

  def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float]

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    var x = context
    x = x + contextMixer(x)
    x = x + crossContextMixer(crossContext, x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

class CrossTransformerLayer[CrossContext: Label, CrossEmbedding: Label, Context: Label, Embedding: Label](
    hyperParams: CrossTransformerLayer.HyperParams[CrossContext, Context, Embedding]
)(
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding]
) extends ICrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]:

  private val selfAttention = MultiHeadSelfAttention(hyperParams.multiHeadAttention)(params.selfAttentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNormParams)

  private val crossAttention = new MultiHeadCrossAttention(hyperParams.multiHeadCrossAttention)(params.crossAttentionParams)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNormParams)

  private val mlp = MLPEmbeddingMixer(hyperParams.embeddingMixer)(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
    mlp(mlpPreNorm(embeddings))

  override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

  override def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    crossAttention(crossContext, context.vmap(Axis[Context])(crossAttentionPreNorm))

object CrossTransformerLayer:

  def apply[CrossContext: Label, Context: Label, CrossEmbedding: Label, Embedding: Label](hyperParams: HyperParams[CrossContext, Context, Embedding])(params: Params[CrossEmbedding, Embedding]): CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding] =
    new CrossTransformerLayer(hyperParams)(params)

  case class HyperParams[CrossContext: Label, Context: Label, Embedding: Label](
      embeddingMixer: MLPEmbeddingMixer.HyperParams[Embedding],
      multiHeadAttention: MultiHeadSelfAttention.HyperParams[Context],
      multiHeadCrossAttention: MultiHeadCrossAttention.HyperParams[CrossContext, Context]
  )

  case class Params[CrossEmbedding, Embedding](
      crossAttentionParams: MultiHeadCrossAttention.Params[CrossEmbedding, Embedding],
      crossAttentionNormParams: LayerNorm.Params[Embedding],
      selfAttentionParams: MultiHeadSelfAttention.Params[Embedding],
      selfAttentionNormParams: LayerNorm.Params[Embedding],
      mlpNormParams: LayerNorm.Params[Embedding],
      mlpParams: MLPEmbeddingMixer.Params[Embedding]
  )

  object Params:

    def xavierUniformDepthScaled[CE: Label, E: Label](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], crossEmbeddingExtent: AxisExtent[CE], embeddingExtent: AxisExtent[E], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], key: Random.Key): Params[CE, E] =
      val (selfAttnKey, crossAttnKey, mixKey) = key.splitToTuple(3)
      new Params[CE, E](
        crossAttentionParams = MultiHeadCrossAttention.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, crossEmbeddingExtent, embeddingExtent, crossAttnKey),
        crossAttentionNormParams = LayerNorm.Params.identity(embeddingExtent),
        selfAttentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, selfAttnKey),
        selfAttentionNormParams = LayerNorm.Params.identity(embeddingExtent),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent)
      )
