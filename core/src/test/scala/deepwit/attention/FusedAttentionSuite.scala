package deepwit.attention

import deepwit.*
import deepwit.base.AffineLayer
import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class FusedAttentionSuite extends AnyFunSpec with Matchers:

  trait Ctx derives Label
  trait Emb derives Label

  private val numHeads = 2
  private val ctxExtent = Axis[Ctx] -> 8
  private val embExtent = Axis[Emb] -> 32 // 16 per head, of which cuDNN needs a multiple of 8

  private def float32Params =
    MultiHeadAttention.Params.xavierUniformDepthScaled(
      numTransformerLayers = 2,
      numHeads = numHeads,
      sourceEmbeddingExtent = embExtent,
      targetEmbeddingExtent = embExtent,
      vtype = VType[Float32],
      key = Random.Key(42)
    )

  /** Converted rather than initialized in place: the initializers answer Float32 either way. */
  private def bfloat16Params =
    val p = float32Params
    MultiHeadAttention.Params(
      queryWeights = p.queryWeights.asFloat(VType[BFloat16]),
      keyWeights = p.keyWeights.asFloat(VType[BFloat16]),
      valueWeights = p.valueWeights.asFloat(VType[BFloat16]),
      outputProjection = AffineLayer.Params(
        weight = p.outputProjection.weight.asFloat(VType[BFloat16]),
        bias = p.outputProjection.bias.asFloat(VType[BFloat16])
      )
    )

  private def context = Normal.standardNormal(Shape(ctxExtent, embExtent)).sample(Random.Key(7))

  private def onCuDnn = assume(FusedMultiHeadAttentionKernel.canRun(DType.BFloat16), "needs cuDNN on a CUDA device")

  describe("MultiHeadFusedCausalAttention"):

    it("computes what the head-by-head formulation computes"):
      onCuDnn
      val p = bfloat16Params
      val x = context.asFloat(VType[BFloat16])
      val fused = MultiHeadFusedCausalAttention(Axis[Ctx], Axis[Ctx], p)
      val reference = MultiHeadCustomAttention(p, causalMask[Ctx, Ctx], AttentionScore.scaledDotProduct)
      fused(x, x) should approxEqual(reference(x, x), 8e-3f)

  describe("MultiHeadFusedFullAttention"):

    it("computes what the head-by-head formulation computes"):
      onCuDnn
      val p = bfloat16Params
      val x = context.asFloat(VType[BFloat16])
      val fused = MultiHeadFusedFullAttention(Axis[Ctx], Axis[Ctx], p)
      val reference = MultiHeadCustomAttention(p, fullMask[Ctx, Ctx], AttentionScore.scaledDotProduct)
      fused(x, x) should approxEqual(reference(x, x), 8e-3f)

    it("attends beyond the diagonal, unlike the causal kernel"):
      onCuDnn
      val p = bfloat16Params
      val x = context.asFloat(VType[BFloat16])
      val full = MultiHeadFusedFullAttention(Axis[Ctx], Axis[Ctx], p)(x, x)
      val causal = MultiHeadFusedCausalAttention(Axis[Ctx], Axis[Ctx], p)(x, x)
      (full.asFloat32 - causal.asFloat32).abs.max.item should be > 1e-2f

  describe("the intermediates of a fused attention"):

    it("come from the kernel's own projections, and agree with the head-by-head formulation"):
      onCuDnn
      val p = bfloat16Params
      val x = context.asFloat(VType[BFloat16])
      val fused = MultiHeadFusedCausalAttention(Axis[Ctx], Axis[Ctx], p)
      val reference = MultiHeadCustomAttention(p, causalMask[Ctx, Ctx], AttentionScore.scaledDotProduct)
      val (attended, intermediates) = fused.applyWithIntermediates(x, x)
      val (referenceAttended, referenceIntermediates) = reference.applyWithIntermediates(x, x)
      attended should approxEqual(referenceAttended, 8e-3f)
      intermediates.queries should approxEqual(referenceIntermediates.queries, 8e-3f)
      intermediates.keys should approxEqual(referenceIntermediates.keys, 8e-3f)
      intermediates.values should approxEqual(referenceIntermediates.values, 8e-3f)

  describe("the fused kernel selection"):

    it("takes causal and full attention alike when cuDNN accepts the parameters"):
      onCuDnn
      MultiHeadCausalAttention(Axis[Ctx], Axis[Ctx], bfloat16Params) shouldBe a[MultiHeadFusedCausalAttention[?, ?, ?, ?]]
      MultiHeadFullAttention(Axis[Ctx], Axis[Ctx], bfloat16Params) shouldBe a[MultiHeadFusedFullAttention[?, ?, ?, ?]]

    it("leaves Float32 on the head-by-head formulation, which cuDNN rejects"):
      MultiHeadCausalAttention(Axis[Ctx], Axis[Ctx], float32Params) should not be a[MultiHeadFusedCausalAttention[?, ?, ?, ?]]
      MultiHeadFullAttention(Axis[Ctx], Axis[Ctx], float32Params) should not be a[MultiHeadFusedFullAttention[?, ?, ?, ?]]
