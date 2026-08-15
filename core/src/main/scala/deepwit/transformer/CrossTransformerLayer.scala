package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import dimwit.Label as Λ
import deepwit.transformer.attention.{MultiHeadSelfAttention, MultiHeadAttention}

/** A single pre-norm transformer layer that additionally attends onto a cross context.
  *
  * The context is mixed along itself (self-attention), then along the cross context
  * (cross-attention), and finally along the embedding.
  *
  * @tparam CrossContext The axis label for the cross sequence.
  * @tparam CrossEmbedding The axis label for the cross embedding space.
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param createCrossAttentionMask A function generating a boolean mask for the cross-attention.
  * @param createSelfAttentionMask A function generating a boolean mask for the self-attention.
  */
class CrossTransformerLayer[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding, V],
    createCrossAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Bool],
    createSelfAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  private val selfAttention = MultiHeadSelfAttention[Context, Embedding, V](params.selfAttentionParams, createSelfAttentionMask)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNormParams)

  private val crossAttention = MultiHeadAttention(params.crossAttentionParams, createCrossAttentionMask)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val contextMixed = context + contextMixer(context)
    val crossContextMixed = contextMixed + crossContextMixer(crossContext, contextMixed)
    crossContextMixed + crossContextMixed.vmap(Axis[Context])(embeddingMixer)

  private def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

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

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[CrossEmbedding, Embedding, V] =
      val (selfAttnKey, crossAttnKey, mixKey) = key.splitToTuple(3)
      new Params[CrossEmbedding, Embedding, V](
        crossAttentionParams = MultiHeadAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, crossEmbeddingExtent, embeddingExtent, vtype, crossAttnKey),
        crossAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        selfAttentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, vtype, selfAttnKey),
        selfAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
