package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import deepwit.attention.{MultiHeadCausalSelfAttention, MultiHeadSelfAttention}
import dimwit.Label as Λ

/** The residual skeleton of a transformer layer.
  *
  * The context is mixed along itself and then along the embedding, each on its own residual branch.
  * What each mixer is remains open to the implementation.
  */
trait TransformerBlock[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  override final def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val contextMixed = context + contextMixer(context)
    contextMixed + contextMixed.vmap(Axis[Context])(embeddingMixer)

  protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V]

  protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V]

/** A single layer of the GPT-2 transformer, mixing along the context and then along the embedding.
  *
  * Every choice here is GPT-2's rather than a theorem: causal self-attention, LayerNorm ahead of
  * both residual branches, and a GELU MLP as the embedding mixer. This is a composed architecture
  * rather than a building block, and is slated to move out of core.
  *
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param contextAxis The axis of the sequence attending onto itself.
  * @param params The learnable parameters.
  */
class CausalTransformerLayer[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: CausalTransformerLayer.Params[Embedding, V]
) extends TransformerBlock[Context, Embedding, V](contextAxis):

  private val selfAttention = MultiHeadCausalSelfAttention(contextAxis, params.attentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.attentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  override protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

object CausalTransformerLayer:

  case class Params[Embedding, V](
      attentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      attentionNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[Embedding, V] =
      val (attnKey, mixKey) = key.splitToTuple(2)
      new Params[Embedding, V](
        attentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, vtype, attnKey),
        attentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
