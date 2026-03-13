package deepwit.embedder

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.{Normal, Uniform}

case class LearnedAbsolutePositionalInjector[Context: Label, Embedding: Label](params: LearnedAbsolutePositionalInjector.Params[Context, Embedding]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    context + params.positionalEmbeddings

object LearnedAbsolutePositionalInjector:

  case class Params[Context, Embedding](positionalEmbeddings: Tensor2[Context, Embedding, Float])

  object Params:

    def lecunUniform[Context: Label, Embedding: Label](contextExtent: AxisExtent[Context], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Context, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      val a = gain * (3f * variance).sqrt
      Params(IndependentDistribution.fromUnivariate(Shape(contextExtent, embeddingExtent), Uniform(-a, a)).sample(key))

    def lecunNormal[Context: Label, Embedding: Label](contextExtent: AxisExtent[Context], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Context, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      Params(Normal.standardIsotropic(Shape(contextExtent, embeddingExtent), scale = gain * variance.sqrt).sample(key))
