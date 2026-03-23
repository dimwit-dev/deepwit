package example.gpt

import dimwit.*
import nn.ActivationFunctions.softmax
import deepwit.*
import deepwit.labels.{Head, HeadKey, HeadQuery, HeadValue}

case class GPT(hyperParams: GPT.HyperParams)(params: GPT.Params):

  private val embedder = VocabularyEmbedder(params.embedderParams)
  private val positionalInjector = LearnedAbsolutePositionalInjector(params.positionalInjectorParams)
  private val causalTransformer = CausalTransformer(hyperParams.transformer)(params.transformer)
  private val outputProjection = LinearLayer(params.outputProjection)

  def logits(tokenContext: Tensor1[Context, Int]): Tensor2[Context, Vocab, Float] =
    val embeddingContext = tokenContext.vmap(Axis[Context])(embedder)
    val sequentialContext = positionalInjector(embeddingContext)
    val mixedContext = causalTransformer(sequentialContext)
    mixedContext.vmap(Axis[Context])(outputProjection)

  def probits(tokenContext: Tensor1[Context, Int]): Tensor2[Context, Vocab, Float] =
    logits(tokenContext)
      .vapply(Axis[Vocab])(softmax)

  def apply(tokenContext: Tensor1[Context, Int]): Tensor1[Context, Int] =
    logits(tokenContext)
      .argmax(Axis[Vocab])

  def generate() = ??? // TODO

object GPT:

  case class HyperParams(
      transformer: Transformer.HyperParams[Context, Embedding]
  )

  case class Params(
      embedderParams: VocabularyEmbedder.Params[Vocab, Embedding],
      positionalInjectorParams: LearnedAbsolutePositionalInjector.Params[Context, Embedding],
      transformer: Transformer.Params[Embedding],
      outputProjection: LinearLayer.Params[Embedding, Vocab]
  )

  object Params:

    def init(numTransformerLayers: Int)(
        vocabExtent: AxisExtent[Vocab],
        contextExtent: AxisExtent[Context],
        headExtent: AxisExtent[Head],
        headQueryExtent: AxisExtent[HeadQuery],
        headKeyExtent: AxisExtent[HeadKey],
        headValueExtent: AxisExtent[HeadValue],
        embeddingExtent: AxisExtent[Embedding],
        embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed],
        key: Random.Key
    ): Params =
      val (vocabEmbeddingKey, positionalEmbeddingKey, transformerKey, outputProjectionKey) = key.splitToTuple(4)

      Params(
        embedderParams = VocabularyEmbedder.Params.lecunUniform(vocabExtent, embeddingExtent, vocabEmbeddingKey),
        positionalInjectorParams = LearnedAbsolutePositionalInjector.Params.lecunUniform(contextExtent, embeddingExtent, positionalEmbeddingKey),
        transformer = Transformer.Params.xavierUniformDepthScaled(numTransformerLayers)(
          headExtent = headExtent,
          headQueryExtent = headQueryExtent,
          headKeyExtent = headKeyExtent,
          headValueExtent = headValueExtent,
          embeddingExtent = embeddingExtent,
          embeddingMixedExtent = embeddingMixedExtent,
          key = transformerKey
        ),
        outputProjection = LinearLayer.Params.xavierUniform(embeddingExtent, vocabExtent, outputProjectionKey)
      )
