package deepwit.transformer.attention

import deepwit.*
import deepwit.transformer.{causalMask, fullMask}
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class MultiHeadSelfAttentionSuite extends AnyFunSpec with Matchers:

  trait Ctx derives Label
  trait Emb derives Label

  private val numHeads = 2
  private val ctxExtent = Axis[Ctx] -> 4
  private val embExtent = Axis[Emb] -> 4

  private def params = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(
    numTransformerLayers = 2,
    numHeads = numHeads,
    embeddingExtent = embExtent,
    vtype = VType[Float32],
    key = Random.Key(42)
  )

  private def context(lastRow: Float) =
    Tensor(Shape(ctxExtent, embExtent)).fromArray(
      Array(
        1f, 2f, 3f, 4f,
        5f, 6f, 7f, 8f,
        9f, 10f, 11f, 12f,
        lastRow, lastRow, lastRow, lastRow
      )
    )

  describe("MultiHeadSelfAttention"):

    it("preserves the shape of the context"):
      val attention = MultiHeadSelfAttention(params, fullMask[Ctx, Ctx])
      val result = attention(context(13f))
      result.shape(Axis[Ctx]) shouldBe 4
      result.shape(Axis[Emb]) shouldBe 4

    it("is multi-head attention of the context onto itself"):
      val p = params
      val selfAttention = MultiHeadSelfAttention(p, fullMask[Ctx, Ctx])
      val crossAttention = MultiHeadAttention(
        MultiHeadAttention.Params(p.queryWeights, p.keyWeights, p.valueWeights, p.outputProjection),
        fullMask[Ctx, Ctx]
      )
      val x = context(13f)
      selfAttention(x) should approxEqual(crossAttention(x, x), 1e-6f)

    it("lets a causal mask hide later positions from earlier ones"):
      val attention = MultiHeadSelfAttention(params, causalMask[Ctx, Ctx])
      val a = attention(context(13f))
      val b = attention(context(-99f))
      // Positions 0 to 2 cannot attend to the perturbed final position.
      a.slice(Axis[Ctx].at(0 until 3)) should approxEqual(b.slice(Axis[Ctx].at(0 until 3)), 1e-5f)
      // The final position does see the change.
      (a.slice(Axis[Ctx].at(3)) - b.slice(Axis[Ctx].at(3))).abs.max.item should be > 1e-3f

    it("lets a full mask expose later positions to earlier ones"):
      val attention = MultiHeadSelfAttention(params, fullMask[Ctx, Ctx])
      val a = attention(context(13f))
      val b = attention(context(-99f))
      (a.slice(Axis[Ctx].at(0)) - b.slice(Axis[Ctx].at(0))).abs.max.item should be > 1e-3f

  describe("MultiHeadSelfAttention.Params.xavierUniformDepthScaled"):

    it("splits the embedding space evenly across the heads"):
      val p = params
      p.queryWeights.shape(Axis[Head]) shouldBe numHeads
      p.queryWeights.shape(Axis[HeadQuery]) shouldBe 2
      p.keyWeights.shape(Axis[HeadKey]) shouldBe 2
      p.valueWeights.shape(Axis[HeadValue]) shouldBe 2
      p.outputProjection.weight.shape(Axis[Head |*| HeadValue]) shouldBe 4
      p.outputProjection.weight.shape(Axis[Emb]) shouldBe 4

    it("requires the embedding size to be divisible by the head count"):
      an[IllegalArgumentException] should be thrownBy
        MultiHeadSelfAttention.Params.xavierUniformDepthScaled(2, 3, embExtent, VType[Float32], Random.Key(42))
