package deepwit.transformer

import dimwit.*
import deepwit.base.ActivationFunction.gelu
import deepwit.normalization.LayerNorm
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue, MultiHeadSelfAttention}

trait ITransformerLayer[Context: Label, Embedding: Label] extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float]
  def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float]

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    var x = context
    x = x + contextMixer(x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

class TransformerLayer[Context: Label, Embedding: Label](
    hyperParams: TransformerLayer.HyperParams[Context, Embedding]
)(
    params: TransformerLayer.Params[Embedding]
) extends ITransformerLayer[Context, Embedding]:

  val selfAttention = MultiHeadSelfAttention(hyperParams.multiHeadAttention)(params.attentionParams)
  val selfAttentionNorm = LayerNorm(params.attentionNormParams)

  val mlp = MLPEmbeddingMixer(hyperParams.embeddingMixer)(params.mlpParams)
  val mlpNorm = LayerNorm(params.mlpNormParams)

  override def embeddingMixer(embeddings: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
    val embNorm = mlpNorm(embeddings)
    mlp(embNorm)

  override def contextMixer(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val contextNorm = context.vmap(Axis[Context])(selfAttentionNorm)
    selfAttention(contextNorm)

object TransformerLayer:

  def apply[Context: Label, Embedding: Label](hyperParams: HyperParams[Context, Embedding])(params: Params[Embedding]): TransformerLayer[Context, Embedding] =
    new TransformerLayer(hyperParams)(params)

  case class HyperParams[Context: Label, Embedding: Label](
      embeddingMixer: MLPEmbeddingMixer.HyperParams[Embedding],
      multiHeadAttention: MultiHeadSelfAttention.HyperParams[Context]
  )

  case class Params[Embedding](
      attentionParams: MultiHeadSelfAttention.Params[Embedding],
      attentionNormParams: LayerNorm.Params[Embedding],
      mlpParams: MLPEmbeddingMixer.Params[Embedding],
      mlpNormParams: LayerNorm.Params[Embedding]
  )

  object Params:

    def xavierUniformDepthScaled[E: Label](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[E], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], key: Random.Key): Params[E] =
      val (attnKey, mixKey) = key.splitToTuple(2)
      new Params[E](
        attentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, attnKey),
        attentionNormParams = LayerNorm.Params.identity(embeddingExtent),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent)
      )
