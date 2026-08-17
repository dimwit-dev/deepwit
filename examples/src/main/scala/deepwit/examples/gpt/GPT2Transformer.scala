package deepwit.examples.gpt

import dimwit.*
import dimwit.Label as Λ

import deepwit.attention.{MultiHeadCausalSelfAttention, MultiHeadSelfAttention}
import deepwit.normalization.LayerNorm
import deepwit.transformer.{EmbeddingMixed, MLPEmbeddingMixer, TransformerBlock}

/** The transformer of GPT-2, as described in *Language Models are Unsupervised Multitask Learners*.
  *
  * A stack of [[GPT2TransformerBlock]]s followed by a final normalization. Composed of deepwit's
  * building blocks rather than provided by them: copy it into your own project and change whatever
  * your architecture asks for.
  *
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param contextAxis The axis of the sequence attending onto itself.
  * @param params The learnable parameters.
  */
class GPT2Transformer[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: GPT2Transformer.Params[Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val blocks = params.transformerBlocks.map(p => GPT2TransformerBlock(contextAxis, p))
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = blocks.foldLeft(context):
      case (context_i, block) => block(context_i)
    res.vmap(Axis[Context])(finalNorm)

object GPT2Transformer:

  case class Params[Embedding, V](
      transformerBlocks: List[GPT2TransformerBlock.Params[Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def init[Embedding: Λ, V: IsFloating](numTransformerBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[Embedding, V] =
      new Params[Embedding, V](
        transformerBlocks =
          key.split(numTransformerBlocks).map: key =>
            GPT2TransformerBlock.Params.init(numTransformerBlocks, numHeads, embeddingExtent, embeddingMixedExtent, vtype, key)
          .toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** A single transformer block of the GPT-2 transformer, mixing along the context and then along the embedding.
  *
  * Every choice here is GPT-2's rather than a theorem: causal self-attention, LayerNorm ahead of
  * both residual branches, and a GELU MLP as the embedding mixer. Swapping in RMSNorm, moving the
  * normalization behind the branches, or replacing the mixer with SwiGLU each gives a different,
  * equally valid architecture — which is why this lives here rather than in core.
  *
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param contextAxis The axis of the sequence attending onto itself.
  * @param params The learnable parameters.
  */
class GPT2TransformerBlock[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: GPT2TransformerBlock.Params[Embedding, V]
) extends TransformerBlock[Context, Embedding, V](contextAxis):

  private val selfAttention = MultiHeadCausalSelfAttention(contextAxis, params.attentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.attentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  override protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

object GPT2TransformerBlock:

  case class Params[Embedding, V](
      attentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      attentionNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def init[Embedding: Λ, V: IsFloating](numTransformerBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[Embedding, V] =
      val (attnKey, mixKey) = key.splitToTuple(2)
      new Params[Embedding, V](
        attentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerBlocks, numHeads, embeddingExtent, vtype, attnKey),
        attentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
