package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import dimwit.Label as Λ
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue, MultiHeadSelfAttention, MultiHeadAttention}

class CrossTransformerLayer[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding, V],
    createCrossAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Bool],
    createSelfAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  private val selfAttention = MultiHeadSelfAttention[Context, Embedding, V](params.selfAttentionParams, createSelfAttentionMask)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNormParams)

  private val crossAttention = new MultiHeadAttention(params.crossAttentionParams, createCrossAttentionMask)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    var x = context
    x = x + contextMixer(x)
    x = x + crossContextMixer(crossContext, x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

  private def embeddingMixer(embeddings: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embeddings))

  private def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

  private def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    crossAttention(crossContext, context.vmap(Axis[Context])(crossAttentionPreNorm))

object CrossTransformerLayer:

  case class Params[CrossEmbedding, Embedding, V](
      crossAttentionParams: MultiHeadAttention.Params[CrossEmbedding, Embedding, V],
      crossAttentionNormParams: LayerNorm.Params[Embedding, V],
      selfAttentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNormParams: LayerNorm.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[CrossEmbedding, Embedding, V] =
      val (selfAttnKey, crossAttnKey, mixKey) = key.splitToTuple(3)
      val headExtent = Axis[Head] -> numHeads
      val headCrossQueryExtent = Axis[HeadQuery] -> crossEmbeddingExtent.size / numHeads
      val headCrossKeyExtent = Axis[HeadKey] -> crossEmbeddingExtent.size / numHeads
      val headCrossValueExtent = Axis[HeadValue] -> crossEmbeddingExtent.size / numHeads
      val headQueryExtent = Axis[HeadQuery] -> embeddingExtent.size / numHeads
      val headKeyExtent = Axis[HeadKey] -> embeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> embeddingExtent.size / numHeads
      new Params[CrossEmbedding, Embedding, V](
        crossAttentionParams = MultiHeadAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, crossEmbeddingExtent, embeddingExtent, vtype, crossAttnKey),
        crossAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        selfAttentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, vtype, selfAttnKey),
        selfAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
