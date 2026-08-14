package example.gpt

import dimwit.*
import dimwit.Conversions.given
import nn.ActivationFunctions.softmax
import deepwit.*
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue}
import dimwit.stats.Categorical
import scala.CanEqual.derived
import deepwit.embedder.VocabularyEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.base.LinearLayer
import deepwit.transformer.Transformer
import deepwit.transformer.MLPEmbeddingMixer

case class GPT[V: IsFloating](params: GPT.Params[V]):

  private val embedder = VocabularyEmbedder(params.embedderParams)
  private val positionalInjector = LearnedAbsolutePositionalInjector(params.positionalInjectorParams)
  private val causalTransformer = Transformer.causal(Axis[Context], params.transformer)
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

  def generate(
      prompt: Seq[Int],
      contextSize: Int = 1024,
      temperature: Float = 1.0f
  )(using key: Random.Key): LazyList[Int] =
    val initialContext = if prompt.size > contextSize then prompt.takeRight(contextSize) else prompt
    val fastLogits = jit(logits)

    LazyList.unfold((initialContext, key)): (currentContext, currentKey) =>

      val padLength = contextSize - currentContext.size
      val padded = currentContext ++ Seq.fill(padLength)(0)

      val contextTensor = Tensor1(Axis[Context], VType[Int32]).fromArray(padded.toArray)
      val stepLogits = fastLogits(contextTensor)
      val lastLogits = stepLogits.slice(Axis[Context].at(currentContext.size - 1))
      val scaledLogits = lastLogits /! temperature
      val probs = Prob(softmax(scaledLogits.asFloat32))

      val (nextKey, sampleKey) = currentKey.split2()
      val nextTokenTensor = Categorical(probs).sample(sampleKey)
      val nextToken = nextTokenTensor.item

      val nextContext = if currentContext.size >= contextSize then
        currentContext.tail :+ nextToken
      else
        currentContext :+ nextToken

      Some((nextToken, (nextContext, nextKey)))

object GPT:

  case class Params[V](
      embedderParams: VocabularyEmbedder.Params[Vocab, Embedding, V],
      positionalInjectorParams: LearnedAbsolutePositionalInjector.Params[Context, Embedding, V],
      transformer: Transformer.Params[Embedding, V],
      outputProjection: LinearLayer.Params[Embedding, Vocab, V]
  )

  object Params:

    given treeBFloat16: TreeOf[GPT.Params[BFloat16], BFloat16] = TreeOf.derived
    given treeFloat32: TreeOf[GPT.Params[Float32], Float32] = TreeOf.derived
    given tensorTreeParamsF16: TensorTree[GPT.Params[BFloat16]] = TensorTree.derived
    given tensorTreeParamsF32: TensorTree[GPT.Params[Float32]] = TensorTree.derived

    def init[V: IsFloating](numTransformerLayers: Int)(
        vocabExtent: AxisExtent[Vocab],
        contextExtent: AxisExtent[Context],
        numHeads: Int,
        embeddingExtent: AxisExtent[Embedding],
        embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed],
        vtype: VType[V],
        key: Random.Key
    ): Params[V] =
      val (vocabEmbeddingKey, positionalEmbeddingKey, transformerKey, outputProjectionKey) = key.splitToTuple(4)

      Params(
        embedderParams = VocabularyEmbedder.Params.lecunUniform(vocabExtent, embeddingExtent, vtype, vocabEmbeddingKey),
        positionalInjectorParams = LearnedAbsolutePositionalInjector.Params.lecunUniform(contextExtent, embeddingExtent, vtype, positionalEmbeddingKey),
        transformer = Transformer.Params.xavierUniformDepthScaled(
          numTransformerLayers,
          numHeads,
          embeddingExtent = embeddingExtent,
          embeddingMixedExtent = embeddingMixedExtent,
          vtype = vtype,
          key = transformerKey
        ),
        outputProjection = LinearLayer.Params.xavierUniform(embeddingExtent, vocabExtent, vtype, outputProjectionKey)
      )
