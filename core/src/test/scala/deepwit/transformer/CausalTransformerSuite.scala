package deepwit.transformer

import deepwit.*
import deepwit.normalization.LayerNorm
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class CausalTransformerSuite extends AnyFunSpec with Matchers:

  trait Ctx derives Label
  trait Emb derives Label

  private val numLayers = 3
  private val ctxExtent = Axis[Ctx] -> 4
  private val embExtent = Axis[Emb] -> 4

  private def params = CausalTransformer.Params.xavierUniformDepthScaled(
    numTransformerLayers = numLayers,
    numHeads = 2,
    embeddingExtent = embExtent,
    embeddingMixedExtent = Axis[EmbeddingMixed] -> 8,
    vtype = VType[Float32],
    key = Random.Key(42)
  )

  private def context(lastRow: Float) =
    Tensor(Shape(ctxExtent, embExtent)).fromArray(
      Array(
        1f, 2f, 3f, 4f,
        5f, 6f, 7f, 8f,
        9f, 10f, 11f, 12f,
        lastRow, 1f, 2f, 3f
      )
    )

  private def firstRowVaried(firstRow: Float) =
    Tensor(Shape(ctxExtent, embExtent)).fromArray(
      Array(
        firstRow, 2f, 3f, 4f,
        5f, 6f, 7f, 8f,
        9f, 10f, 11f, 12f,
        13f, 1f, 2f, 3f
      )
    )

  describe("CausalTransformer"):

    it("preserves the shape of the context"):
      val result = CausalTransformer(Axis[Ctx], params)(context(13f))
      result.shape(Axis[Ctx]) shouldBe 4
      result.shape(Axis[Emb]) shouldBe 4

    it("is just the final normalization when it has no layers"):
      val normParams = LayerNorm.Params.identity(embExtent, VType[Float32])
      val transformer = CausalTransformer(Axis[Ctx], CausalTransformer.Params(List.empty, normParams))
      val x = context(13f)
      transformer(x) should approxEqual(x.vmap(Axis[Ctx])(LayerNorm(normParams)), 1e-5f)

    it("keeps earlier positions independent of later ones, being causal"):
      val transformer = CausalTransformer(Axis[Ctx], params)
      val a = transformer(context(13f))
      val b = transformer(context(-99f))
      a.slice(Axis[Ctx].at(0 until 3)) should approxEqual(b.slice(Axis[Ctx].at(0 until 3)), 1e-4f)
      (a.slice(Axis[Ctx].at(3)) - b.slice(Axis[Ctx].at(3))).abs.max.item should be > 1e-3f

    it("lets later positions see the earlier ones"):
      val transformer = CausalTransformer(Axis[Ctx], params)
      val a = transformer(firstRowVaried(1f))
      val b = transformer(firstRowVaried(-99f))
      (a.slice(Axis[Ctx].at(3)) - b.slice(Axis[Ctx].at(3))).abs.max.item should be > 1e-3f

  describe("CausalTransformer.Params.xavierUniformDepthScaled"):

    it("builds one parameter set per layer plus the final normalization"):
      val p = params
      p.transformerLayers.size shouldBe numLayers
      p.finalNorm.weight should approxEqual(Tensor(Shape1(embExtent)).fill(1f), 1e-6f)
      p.finalNorm.bias should approxEqual(Tensor(Shape1(embExtent)).fill(0f), 1e-6f)

    it("gives each layer its own parameters"):
      val p = params
      val first = p.transformerLayers.head.attentionParams.queryWeights
      val second = p.transformerLayers(1).attentionParams.queryWeights
      (first - second).abs.max.item should be > 0f
