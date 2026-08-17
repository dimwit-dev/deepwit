package deepwit.transformer

import deepwit.*
import deepwit.base.AffineLayer
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class CausalTransformerLayerSuite extends AnyFunSpec with Matchers:

  trait Ctx derives Label
  trait Emb derives Label

  private val ctxExtent = Axis[Ctx] -> 4
  private val embExtent = Axis[Emb] -> 4

  private def params = CausalTransformerLayer.Params.xavierUniformDepthScaled(
    numTransformerLayers = 2,
    numHeads = 2,
    embeddingExtent = embExtent,
    embeddingMixedExtent = Axis[EmbeddingMixed] -> 8,
    vtype = VType[Float32],
    key = Random.Key(42)
  )

  /** Zeroes both residual branches, leaving only the identity paths. */
  private def zeroedBranches(p: CausalTransformerLayer.Params[Emb, Float32]) =
    def zeroAffine[In: Label, Out: Label](a: AffineLayer.Params[In, Out, Float32]) =
      AffineLayer.Params(Tensor.like(a.weight).fill(0f), Tensor.like(a.bias).fill(0f))
    p.copy(
      attentionParams = p.attentionParams.copy(outputProjection = zeroAffine(p.attentionParams.outputProjection)),
      mlpParams = p.mlpParams.copy(project = zeroAffine(p.mlpParams.project))
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

  describe("CausalTransformerLayer"):

    it("preserves the shape of the context"):
      val layer = CausalTransformerLayer(Axis[Ctx], params)
      val result = layer(context(13f))
      result.shape(Axis[Ctx]) shouldBe 4
      result.shape(Axis[Emb]) shouldBe 4

    it("is the identity when both residual branches are zeroed"):
      val layer = CausalTransformerLayer(Axis[Ctx], zeroedBranches(params))
      val x = context(13f)
      layer(x) should approxEqual(x, 1e-5f)

    it("changes the context when the branches are not zeroed"):
      val layer = CausalTransformerLayer(Axis[Ctx], params)
      val x = context(13f)
      (layer(x) - x).abs.max.item should be > 1e-3f

    it("keeps earlier positions independent of later ones, being causal"):
      val layer = CausalTransformerLayer(Axis[Ctx], params)
      val a = layer(context(13f))
      val b = layer(context(-99f))
      a.slice(Axis[Ctx].at(0 until 3)) should approxEqual(b.slice(Axis[Ctx].at(0 until 3)), 1e-4f)
      (a.slice(Axis[Ctx].at(3)) - b.slice(Axis[Ctx].at(3))).abs.max.item should be > 1e-3f

  describe("CausalTransformerLayer.Params.xavierUniformDepthScaled"):

    it("builds attention, mixer and both pre-norms"):
      val p = params
      p.attentionParams.queryWeights.shape(Axis[Emb]) shouldBe 4
      p.mlpParams.expand.weight.shape(Axis[EmbeddingMixed]) shouldBe 8
      p.attentionNormParams.weight should approxEqual(Tensor(Shape1(embExtent)).fill(1f), 1e-6f)
      p.attentionNormParams.bias should approxEqual(Tensor(Shape1(embExtent)).fill(0f), 1e-6f)
      p.mlpNormParams.weight should approxEqual(Tensor(Shape1(embExtent)).fill(1f), 1e-6f)
