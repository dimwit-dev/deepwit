package deepwit.transformer

import deepwit.*
import deepwit.attention.{causalMask, fullMask}
import deepwit.base.AffineLayer
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class CrossTransformerLayerSuite extends AnyFunSpec with Matchers:

  trait CrossCtx derives Label
  trait CrossEmb derives Label
  trait Ctx derives Label
  trait Emb derives Label

  private val crossCtxExtent = Axis[CrossCtx] -> 3
  private val ctxExtent = Axis[Ctx] -> 4
  private val embExtent = Axis[Emb] -> 4

  private def params(crossEmbeddingExtent: AxisExtent[CrossEmb] = Axis[CrossEmb] -> 4) =
    CrossTransformerLayer.Params.xavierUniformDepthScaled(
      numTransformerLayers = 2,
      numHeads = 2,
      crossEmbeddingExtent = crossEmbeddingExtent,
      embeddingExtent = embExtent,
      embeddingMixedExtent = Axis[EmbeddingMixed] -> 8,
      vtype = VType[Float32],
      key = Random.Key(42)
    )

  private def crossContext(embeddingSize: Int = 4) =
    Tensor(Shape(crossCtxExtent, Axis[CrossEmb] -> embeddingSize), VType[Float32]).fill(0.5f)

  private def context(lastRow: Float) =
    Tensor(Shape(ctxExtent, embExtent)).fromArray(
      Array(
        1f, 2f, 3f, 4f,
        5f, 6f, 7f, 8f,
        9f, 10f, 11f, 12f,
        lastRow, 1f, 2f, 3f
      )
    )

  /** Zeroes all three residual branches, leaving only the identity paths. */
  private def zeroedBranches(p: CrossTransformerLayer.Params[CrossEmb, Emb, Float32]) =
    def zeroAffine[In: Label, Out: Label](a: AffineLayer.Params[In, Out, Float32]) =
      AffineLayer.Params(Tensor.like(a.weight).fill(0f), Tensor.like(a.bias).fill(0f))
    p.copy(
      selfAttentionParams = p.selfAttentionParams.copy(outputProjection = zeroAffine(p.selfAttentionParams.outputProjection)),
      crossAttentionParams = p.crossAttentionParams.copy(outputProjection = zeroAffine(p.crossAttentionParams.outputProjection)),
      mlpParams = p.mlpParams.copy(project = zeroAffine(p.mlpParams.project))
    )

  describe("CrossTransformerLayer"):

    it("preserves the shape of the context"):
      val layer = CrossTransformerLayer(params(), fullMask[Ctx, CrossCtx], fullMask[Ctx, Ctx])
      val result = layer(crossContext(), context(13f))
      result.shape(Axis[Ctx]) shouldBe 4
      result.shape(Axis[Emb]) shouldBe 4

    it("is the identity when all three residual branches are zeroed"):
      val layer = CrossTransformerLayer(zeroedBranches(params()), fullMask[Ctx, CrossCtx], fullMask[Ctx, Ctx])
      val x = context(13f)
      layer(crossContext(), x) should approxEqual(x, 1e-5f)

    it("responds to the cross context"):
      val layer = CrossTransformerLayer(params(), fullMask[Ctx, CrossCtx], fullMask[Ctx, Ctx])
      val a = layer(Tensor(Shape(crossCtxExtent, Axis[CrossEmb] -> 4), VType[Float32]).fill(0.5f), context(13f))
      val b = layer(Tensor(Shape(crossCtxExtent, Axis[CrossEmb] -> 4), VType[Float32]).fill(-2f), context(13f))
      (a - b).abs.max.item should be > 1e-3f

    it("attends onto a cross context whose embedding space differs from its own"):
      val layer = CrossTransformerLayer(params(Axis[CrossEmb] -> 6), fullMask[Ctx, CrossCtx], fullMask[Ctx, Ctx])
      val result = layer(crossContext(embeddingSize = 6), context(13f))
      result.shape(Axis[Ctx]) shouldBe 4
      result.shape(Axis[Emb]) shouldBe 4

    it("keeps earlier positions independent of later ones under a causal self-attention mask"):
      val layer = CrossTransformerLayer(params(), fullMask[Ctx, CrossCtx], causalMask[Ctx, Ctx])
      val a = layer(crossContext(), context(13f))
      val b = layer(crossContext(), context(-99f))
      a.slice(Axis[Ctx].at(0 until 3)) should approxEqual(b.slice(Axis[Ctx].at(0 until 3)), 1e-4f)
      (a.slice(Axis[Ctx].at(3)) - b.slice(Axis[Ctx].at(3))).abs.max.item should be > 1e-3f

  describe("CrossTransformerLayer.Params.xavierUniformDepthScaled"):

    it("shapes the cross attention from the cross embedding and the self attention from its own"):
      val p = params(Axis[CrossEmb] -> 6)
      p.crossAttentionParams.keyWeights.shape(Axis[CrossEmb]) shouldBe 6
      p.crossAttentionParams.queryWeights.shape(Axis[Emb]) shouldBe 4
      p.selfAttentionParams.queryWeights.shape(Axis[Emb]) shouldBe 4
      p.mlpParams.expand.weight.shape(Axis[EmbeddingMixed]) shouldBe 8
