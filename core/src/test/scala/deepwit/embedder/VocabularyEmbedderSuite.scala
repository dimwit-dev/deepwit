package deepwit.embedder

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class VocabularyEmbedderSuite extends AnyFunSpec with Matchers:

  private val vocabExtent = Axis[A] -> 3
  private val embeddingExtent = Axis[B] -> 2

  // format: off
  private val embeddings = Tensor(Shape(vocabExtent, embeddingExtent)).fromArray(Array[Float](
    1f, 2f,
    3f, 4f,
    5f, 6f
  ))
  // format: on

  private val embedder = VocabularyEmbedder(VocabularyEmbedder.Params(embeddings))

  describe("VocabularyEmbedder"):

    it("looks up the embedding row of a token"):
      embedder(Tensor0(1)) should approxEqual(Tensor(Shape1(embeddingExtent)).fromArray(Array(3f, 4f)), 1e-6f)
      embedder(Tensor0(2)) should approxEqual(Tensor(Shape1(embeddingExtent)).fromArray(Array(5f, 6f)), 1e-6f)

    it("unembeds by projecting onto every vocabulary embedding"):
      val embedding = Tensor(Shape1(embeddingExtent)).fromArray(Array(1f, 0f))
      // dot with each row: (1, 3, 5)
      embedder.unembed(embedding) should approxEqual(Tensor(Shape1(vocabExtent)).fromArray(Array(1f, 3f, 5f)), 1e-6f)

    it("recovers the token through unembed when the embeddings are orthonormal"):
      val orthonormal = VocabularyEmbedder(VocabularyEmbedder.Params(Tensor2.eye(vocabExtent, VType[Float32])))
      (0 until 3).foreach: token =>
        val logits = orthonormal.unembed(orthonormal(Tensor0(token)))
        logits.argmax(Axis[A]).item shouldBe token

  describe("VocabularyEmbedder.Params"):

    it("lecunUniform has the requested shape and stays within its bounds"):
      // a = sqrt(3 / embeddingDim) = sqrt(3 / 2)
      val params = VocabularyEmbedder.Params.lecunUniform(Axis[A] -> 8, Axis[B] -> 2, VType[Float32], Random.Key(42))
      params.vocabularyEmbeddings.shape(Axis[A]) shouldBe 8
      params.vocabularyEmbeddings.shape(Axis[B]) shouldBe 2
      params.vocabularyEmbeddings.abs.max.item should be <= math.sqrt(1.5).toFloat

    it("lecunNormal has the requested shape"):
      val params = VocabularyEmbedder.Params.lecunNormal(Axis[A] -> 8, Axis[B] -> 4, VType[Float32], Random.Key(42))
      params.vocabularyEmbeddings.shape(Axis[A]) shouldBe 8
      params.vocabularyEmbeddings.shape(Axis[B]) shouldBe 4
