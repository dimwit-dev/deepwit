package deepwit.transformer

import deepwit.*
import deepwit.activation.gelu
import deepwit.base.AffineLayer
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class MLPEmbeddingMixerSuite extends AnyFunSpec with Matchers:

  trait Emb derives Label

  private val embExtent = Axis[Emb] -> 4
  private val mixedExtent = Axis[EmbeddingMixed] -> 8

  private def params = MLPEmbeddingMixer.Params.xavierUniform(embExtent, mixedExtent, VType[Float32], Random.Key(42))

  private def embedding = Tensor(Shape1(embExtent)).fromArray(Array(1f, -2f, 0.5f, 3f))

  describe("MLPEmbeddingMixer"):

    it("returns the embedding space it was given"):
      MLPEmbeddingMixer(params)(embedding).shape(Axis[Emb]) shouldBe 4

    it("expands through gelu and projects back by default"):
      val p = params
      val expected = AffineLayer(p.project)(gelu(AffineLayer(p.expand)(embedding)))
      MLPEmbeddingMixer(p)(embedding) should approxEqual(expected, 1e-6f)

    it("honours a custom activation"):
      val p = params
      val identityActivation: Tensor1[EmbeddingMixed, Float32] => Tensor1[EmbeddingMixed, Float32] = x => x
      val expected = AffineLayer(p.project)(AffineLayer(p.expand)(embedding))
      MLPEmbeddingMixer(p, identityActivation)(embedding) should approxEqual(expected, 1e-6f)

    it("differs from the default when the activation changes"):
      val p = params
      val identityActivation: Tensor1[EmbeddingMixed, Float32] => Tensor1[EmbeddingMixed, Float32] = x => x
      (MLPEmbeddingMixer(p)(embedding) - MLPEmbeddingMixer(p, identityActivation)(embedding)).abs.max.item should be > 1e-3f

  describe("MLPEmbeddingMixer.Params"):

    it("xavierUniform widens into the mixed space and projects back"):
      val p = params
      p.expand.weight.shape(Axis[Emb]) shouldBe 4
      p.expand.weight.shape(Axis[EmbeddingMixed]) shouldBe 8
      p.project.weight.shape(Axis[EmbeddingMixed]) shouldBe 8
      p.project.weight.shape(Axis[Emb]) shouldBe 4

    it("xavierNormal has the same shapes"):
      val p = MLPEmbeddingMixer.Params.xavierNormal(embExtent, mixedExtent, VType[Float32], Random.Key(42))
      p.expand.weight.shape(Axis[EmbeddingMixed]) shouldBe 8
      p.project.weight.shape(Axis[Emb]) shouldBe 4
