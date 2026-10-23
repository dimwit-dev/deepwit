package deepwit.examples.gpt

import dimwit.*
import dimwit.Label as Λ

import deepwit.attention.{KVCache, MultiHeadCausalSelfAttention, MultiHeadSelfAttention}
import deepwit.normalization.LayerNorm
import deepwit.transformer.TransformerBlock

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

  /** Transforms `context` like [[apply]], and keeps one [[KVCache]] per block to [[decode]] the positions after it. */
  def prefill(context: Tensor2[Context, Embedding, V]): (Tensor2[Context, Embedding, V], List[KVCache[Context, V]]) =
    val (res, caches) = blocks.foldLeft((context, List.empty[KVCache[Context, V]])):
      case ((context_i, caches), block) =>
        val (next, cache) = block.prefill(context_i)
        (next, caches :+ cache)
    (res.vmap(Axis[Context])(finalNorm), caches)

  /** Transforms the `embedding` at `position`, as [[apply]] would, given one [[KVCache]] per block. */
  def decode(caches: List[KVCache[Context, V]], position: Tensor0[Int32], embedding: Tensor1[Embedding, V]): (Tensor1[Embedding, V], List[KVCache[Context, V]]) =
    val (res, nextCaches) = blocks.zip(caches).foldLeft((embedding, List.empty[KVCache[Context, V]])):
      case ((embedding_i, nextCaches), (block, cache)) =>
        val (next, nextCache) = block.decode(cache, position, embedding_i)
        (next, nextCaches :+ nextCache)
    (finalNorm(res), nextCaches)

object GPT2Transformer:

  case class Params[Embedding, V](
      transformerBlocks: List[GPT2TransformerBlock.Params[Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def gpt2Init[Embedding: Λ, V: IsFloating](numTransformerBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      new Params[Embedding, V](
        transformerBlocks =
          key.split(numTransformerBlocks).map: key =>
            GPT2TransformerBlock.Params.gpt2Init(numTransformerBlocks, numHeads, embeddingExtent, embeddingMixedExtent, key, vtype)
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

  /** Applies the block like [[apply]], and keeps the [[KVCache]] of its attention. */
  def prefill(context: Tensor2[Context, Embedding, V]): (Tensor2[Context, Embedding, V], KVCache[Context, V]) =
    val (attended, cache) = selfAttention.prefill(context.vmap(Axis[Context])(selfAttentionPreNorm))
    val contextMixed = context + attended
    (contextMixed + contextMixed.vmap(Axis[Context])(embeddingMixer), cache)

  /** Applies the block to the `embedding` at `position`, as [[apply]] would, attending to the positions in `cache`. */
  def decode(cache: KVCache[Context, V], position: Tensor0[Int32], embedding: Tensor1[Embedding, V]): (Tensor1[Embedding, V], KVCache[Context, V]) =
    val (attended, nextCache) = selfAttention.decode(cache, position, selfAttentionPreNorm(embedding))
    val contextMixed = embedding + attended
    (contextMixed + embeddingMixer(contextMixed), nextCache)

object GPT2TransformerBlock:

  case class Params[Embedding, V](
      attentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      attentionNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def gpt2Init[Embedding: Λ, V: IsFloating](numTransformerBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (attnKey, mixKey) = key.splitToTuple(2)
      new Params[Embedding, V](
        attentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numTransformerBlocks, numHeads, embeddingExtent, attnKey, vtype),
        attentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mixKey, vtype),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
