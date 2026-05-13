package deepwit.embedder

import dimwit.*
import dimwit.stats.{Normal, Uniform}

case class LearnedAbsolutePositionalInjector[Context: Label, Embedding: Label, V: IsFloating](params: LearnedAbsolutePositionalInjector.Params[Context, Embedding, V]) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    context + params.positionalEmbeddings

object LearnedAbsolutePositionalInjector:

  case class Params[Context, Embedding, V](positionalEmbeddings: Tensor2[Context, Embedding, V])

  object Params:

    def lecunUniform[Context: Label, Embedding: Label, V: IsFloating](contextExtent: AxisExtent[Context], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key, gain: Float = 1.0): Params[Context, Embedding, V] =
      val variance = Tensor0(vtype)(1.0f / embeddingExtent.size)
      val a = gain * (3f * variance).sqrt
      Params(IndependentDistribution.fromUnivariate(Shape(contextExtent, embeddingExtent), Uniform(-a, a)).sample(key))

    def lecunNormal[Context: Label, Embedding: Label, V: IsFloating](contextExtent: AxisExtent[Context], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key, gain: Float = 1.0): Params[Context, Embedding, V] =
      val variance = Tensor0(vtype)(1.0f / embeddingExtent.size)
      Params(Normal.standardIsotropic(Shape(contextExtent, embeddingExtent), scale = gain * variance.sqrt).sample(key))
