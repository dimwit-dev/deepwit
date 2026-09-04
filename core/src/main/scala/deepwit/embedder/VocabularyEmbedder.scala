package deepwit.embedder

import dimwit.*
import dimwit.stats.Normal
import dimwit.stats.Uniform
import dimwit.Label as Λ

/** Maps a token to its embedding. [[VocabularyEmbedder.unembed]] maps back (weight typing). */
class VocabularyEmbedder[Vocab: Λ, Embedding: Λ, V: IsFloating](params: VocabularyEmbedder.Params[Vocab, Embedding, V]) extends (Tensor0[Int32] => Tensor1[Embedding, V]):

  override def apply(token: Tensor0[Int32]): Tensor1[Embedding, V] =
    params.vocabularyEmbeddings.slice(Axis[Vocab].at(token))

  /** Scores every token by projecting onto its embedding, through the matrix the lookup uses —
    * weight tying, as described in [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859).
    */
  def unembed(embedding: Tensor1[Embedding, V]): Tensor1[Vocab, V] =
    embedding.dot(Axis[Embedding])(params.vocabularyEmbeddings)

object VocabularyEmbedder:

  case class Params[Vocab, Embedding, V](vocabularyEmbeddings: Tensor2[Vocab, Embedding, V])

  /** Scaled by the embedding size — a lookup has no fan-in of its own, and this leaves each row
    * near unit norm.
    */
  object Params:

    def init[Vocab: Λ, Embedding: Λ, V: IsFloating](vocabExtent: AxisExtent[Vocab], embeddingExtent: AxisExtent[Embedding], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1.0): Params[Vocab, Embedding, V] =
      lecunNormal(vocabExtent, embeddingExtent, key, vtype, gain)

    def lecunUniform[Vocab: Λ, Embedding: Λ, V: IsFloating](vocabExtent: AxisExtent[Vocab], embeddingExtent: AxisExtent[Embedding], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1.0): Params[Vocab, Embedding, V] =
      val variance = Tensor0(vtype)(1.0f / embeddingExtent.size)
      val a = gain * (3f * variance).sqrt
      Params(IndependentDistribution.fromUnivariate(Shape(vocabExtent, embeddingExtent), Uniform(-a, a)).sample(key))

    def lecunNormal[Vocab: Λ, Embedding: Λ, V: IsFloating](vocabExtent: AxisExtent[Vocab], embeddingExtent: AxisExtent[Embedding], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1.0): Params[Vocab, Embedding, V] =
      val variance = Tensor0(vtype)(1.0f / embeddingExtent.size)
      Params(Normal.standardIsotropic(Shape(vocabExtent, embeddingExtent), scale = gain * variance.sqrt).sample(key))
