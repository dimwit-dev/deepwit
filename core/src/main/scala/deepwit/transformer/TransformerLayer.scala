package deepwit.transformer

import dimwit.*
import deepwit.base.gelu
import deepwit.normalization.LayerNorm
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue, MultiHeadSelfAttention}
import dimwit.Label as Λ

class TransformerLayer[Context: Λ, Embedding: Λ, V: IsFloating](
    params: TransformerLayer.Params[Embedding, V],
    createAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val selfAttention = MultiHeadSelfAttention(params.attentionParams, createAttentionMask)
  private val selfAttentionNorm = LayerNorm(params.attentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpNorm = LayerNorm(params.mlpNormParams)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    var x = context
    x = x + contextMixer(x)
    x = x + x.vmap(Axis[Context])(embeddingMixer)
    x

  private def embeddingMixer(embeddings: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpNorm(embeddings))

  private def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionNorm))

object TransformerLayer:

  case class Params[Embedding, V](
      attentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      attentionNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (attnKey, mixKey) = key.splitToTuple(2)
      new Params[Embedding, V](
        attentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, vtype, attnKey),
        attentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
