package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import dimwit.Label as Λ
import deepwit.attention.{MultiHeadAttention, MultiHeadSelfAttention, MultiHeadFullAttention, MultiHeadFullSelfAttention}

/** The residual skeleton of a transformer layer that additionally attends onto a cross context.
  *
  * The context is mixed along itself, then along the cross context, and finally along the embedding,
  * each on its own residual branch. What each mixer is remains open to the implementation.
  */
trait CrossTransformerBlock[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    crossContextAxis: Axis[CrossContext],
    contextAxis: Axis[Context]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  override final def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val contextMixed = context + contextMixer(context)
    val crossContextMixed = contextMixed + crossContextMixer(crossContext, contextMixed)
    crossContextMixed + crossContextMixed.vmap(Axis[Context])(embeddingMixer)

  protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V]

  protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V]

  protected def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V]

/** A single pre-norm transformer layer that additionally attends onto a cross context.
  *
  * The context is mixed along itself (self-attention), then along the cross context
  * (cross-attention), and finally along the embedding. Attention is unrestricted in both
  * directions, which suits a context that is a set rather than a sequence — the object queries of a
  * detection model, say, where every position has to see every other one to settle what it stands
  * for. This is a composed architecture rather than a building block, and is slated to move out of
  * core.
  *
  * @tparam CrossContext The axis label for the cross sequence.
  * @tparam CrossEmbedding The axis label for the cross embedding space.
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param crossContextAxis The axis of the sequence being attended onto.
  * @param contextAxis The axis of the sequence attending onto itself and onto the cross context.
  * @param params The learnable parameters.
  */
class CrossTransformerLayer[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    crossContextAxis: Axis[CrossContext],
    contextAxis: Axis[Context],
    params: CrossTransformerLayer.Params[CrossEmbedding, Embedding, V]
) extends CrossTransformerBlock[CrossContext, CrossEmbedding, Context, Embedding, V](crossContextAxis, contextAxis):

  private val selfAttention = MultiHeadFullSelfAttention[Context, Embedding, V](contextAxis, params.selfAttentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNormParams)

  private val crossAttention = MultiHeadFullAttention(crossContextAxis, contextAxis, params.crossAttentionParams)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  override protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

  override protected def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
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

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[CrossEmbedding, Embedding, V] =
      val (selfAttnKey, crossAttnKey, mixKey) = key.splitToTuple(3)
      new Params[CrossEmbedding, Embedding, V](
        crossAttentionParams = MultiHeadAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, crossEmbeddingExtent, embeddingExtent, vtype, crossAttnKey),
        crossAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        selfAttentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, vtype, selfAttnKey),
        selfAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
