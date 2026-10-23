package deepwit.examples.gpt

import dimwit.*
import deepwit.activation.softmax
import deepwit.attention.{Head, HeadKey, HeadQuery, HeadValue, KVCache}
import dimwit.stats.Categorical
import deepwit.embedder.VocabularyEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.base.LinearLayer

class GPT[V: IsFloating](params: GPT.Params[V]):

  private val embedder = VocabularyEmbedder(params.embedderParams)
  private val positionalInjector = LearnedAbsolutePositionalInjector(params.positionalInjectorParams)
  private val causalTransformer = GPT2Transformer(Axis[Context], params.transformer)
  private val outputProjection = LinearLayer(params.outputProjection)

  def logits(tokenContext: Tensor1[Context, Int32]): Tensor2[Context, Vocab, V] =
    val embeddingContext = tokenContext.vmap(Axis[Context])(embedder)
    val sequentialContext = positionalInjector(embeddingContext)
    val mixedContext = causalTransformer(sequentialContext)
    mixedContext.vmap(Axis[Context])(outputProjection)
    // mixedContext.vmap(Axis[Context])(embedder.unembed) // weight tying

  def probits(tokenContext: Tensor1[Context, Int32]): Tensor2[Context, Vocab, V] =
    logits(tokenContext)
      .vapply(Axis[Vocab])(softmax)

  def apply(tokenContext: Tensor1[Context, Int32]): Tensor1[Context, Int32] =
    logits(tokenContext)
      .argmax(Axis[Vocab])

  /** The [[KVCache]]s of a prompt, which fills `tokenContext` from its start, to [[decode]] the positions after it. */
  def prefill(tokenContext: Tensor1[Context, Int32]): List[KVCache[Context, V]] =
    val embeddingContext = tokenContext.vmap(Axis[Context])(embedder)
    val (_, caches) = causalTransformer.prefill(positionalInjector(embeddingContext))
    caches

  /** The logits for the token after `token` at `position`, as [[logits]] would compute them, given the caches of the positions before. */
  def decode(caches: List[KVCache[Context, V]], position: Tensor0[Int32], token: Tensor0[Int32]): (Tensor1[Vocab, V], List[KVCache[Context, V]]) =
    val embedding = positionalInjector.injectAt(position)(embedder(token))
    val (mixed, nextCaches) = causalTransformer.decode(caches, position, embedding)
    (outputProjection(mixed), nextCaches)

  /** Samples `numTokens` tokens following `prompt`, decoding one position at a time with [[KVCache]]s.
    *
    * The whole loop runs as one `scan` on the device. The last token of the prompt is decoded once more, so that
    * every step samples from the logits it decoded itself.
    */
  def generate(prompt: Seq[Int], numTokens: Int, temperature: Float = 1.0f)(using key: Key): Seq[Int] =
    val contextSize = params.positionalInjectorParams.positionalEmbeddings.shape(Axis[Context])
    require(prompt.nonEmpty, "Generating needs a prompt of at least one token.")
    require(prompt.size + numTokens <= contextSize, s"A prompt of ${prompt.size} tokens leaves room for ${contextSize - prompt.size} more, not $numTokens.")

    val caches = prefill(Tensor1(Axis[Context], VType[Int32]).fromArray(prompt.padTo(contextSize, 0).toArray))
    val positions = Tensor(Shape1(Axis[Context] -> numTokens)).fromArray(Array.range(prompt.size - 1, prompt.size - 1 + numTokens))
    val (_, tokens) = scan(Axis[Context])((caches, Tensor0(prompt.last), key), positions):
      case ((caches, token, key), position) =>
        val (logits, nextCaches) = decode(caches, position, token)
        val (nextKey, sampleKey) = key.split2()
        val nextToken = Categorical(Prob(softmax(logits.asFloat32 /! temperature))).sample(sampleKey)
        ((nextCaches, nextToken, nextKey), nextToken)
    tokens.toArray.toSeq

object GPT:

  case class Params[V](
      embedderParams: VocabularyEmbedder.Params[Vocab, Embedding, V],
      positionalInjectorParams: LearnedAbsolutePositionalInjector.Params[Context, Embedding, V],
      transformer: GPT2Transformer.Params[Embedding, V],
      outputProjection: LinearLayer.Params[Embedding, Vocab, V]
  )

  object Params:

    given treeBFloat16: TreeOf[GPT.Params[BFloat16], BFloat16] = TreeOf.derived
    given treeFloat32: TreeOf[GPT.Params[Float32], Float32] = TreeOf.derived
    given tensorTreeParamsF16: TensorTree[GPT.Params[BFloat16]] = TensorTree.derived
    given tensorTreeParamsF32: TensorTree[GPT.Params[Float32]] = TensorTree.derived

    def gpt2Init[V: IsFloating](numTransformerLayers: Int)(
        vocabExtent: AxisExtent[Vocab],
        contextExtent: AxisExtent[Context],
        numHeads: Int,
        embeddingExtent: AxisExtent[Embedding],
        embeddingMixedExtent: AxisExtent[EmbeddingMixed],
        key: Key,
        vtype: VType[V] = VType[Float32]
    ): Params[V] =
      val (vocabEmbeddingKey, positionalEmbeddingKey, transformerKey, outputProjectionKey) = key.splitToTuple(4)

      Params(
        embedderParams = VocabularyEmbedder.Params.lecunUniform(vocabExtent, embeddingExtent, vocabEmbeddingKey, vtype),
        positionalInjectorParams = LearnedAbsolutePositionalInjector.Params.lecunUniform(contextExtent, embeddingExtent, positionalEmbeddingKey, vtype),
        transformer = GPT2Transformer.Params.gpt2Init(numTransformerLayers, numHeads, embeddingExtent, embeddingMixedExtent, transformerKey, vtype),
        outputProjection = LinearLayer.Params.xavierUniform(embeddingExtent, vocabExtent, outputProjectionKey, vtype)
      )
