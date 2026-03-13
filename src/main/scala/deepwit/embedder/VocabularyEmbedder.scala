package deepwit.embedder

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal
import dimwit.stats.Uniform

case class VocabularyEmbedder[Vocab: Label, Embedding: Label](params: VocabularyEmbedder.Params[Vocab, Embedding]) extends (Tensor0[Int] => Tensor1[Embedding, Float]):

  override def apply(token: Tensor0[Int]): Tensor1[Embedding, Float] =
    params.vocabularyEmbeddings.slice(Axis[Vocab].at(token))

object VocabularyEmbedder:

  case class Params[Vocab, Embedding](vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float])

  object Params:

    def lecunUniform[Vocab: Label, Embedding: Label](vocabExtent: AxisExtent[Vocab], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Vocab, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      val a = gain * (3f * variance).sqrt
      Params(IndependentDistribution.fromUnivariate(Shape(vocabExtent, embeddingExtent), Uniform(-a, a)).sample(key))

    def lecunNormal[Vocab: Label, Embedding: Label](vocabExtent: AxisExtent[Vocab], embeddingExtent: AxisExtent[Embedding], key: Random.Key, gain: Float = 1.0): Params[Vocab, Embedding] =
      val variance = Tensor0(1.0f / embeddingExtent.size)
      Params(Normal.standardIsotropic(Shape(vocabExtent, embeddingExtent), scale = gain * variance.sqrt).sample(key))
