package deepwit.attention

import deepwit.*
import deepwit.base.AffineLayer
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class MultiHeadAttentionSuite extends AnyFunSpec with Matchers:

  trait Src derives Label
  trait SrcEmb derives Label
  trait Tgt derives Label
  trait TgtEmb derives Label

  private val numHeads = 2
  private val srcExtent = Axis[Src] -> 3
  private val tgtExtent = Axis[Tgt] -> 3
  private val embExtent = Axis[TgtEmb] -> 4

  private def params(sourceEmbeddingExtent: AxisExtent[SrcEmb] = Axis[SrcEmb] -> 4) =
    MultiHeadAttention.Params.xavierUniformDepthScaled(
      numTransformerLayers = 2,
      numHeads = numHeads,
      sourceEmbeddingExtent = sourceEmbeddingExtent,
      targetEmbeddingExtent = embExtent,
      vtype = VType[Float32],
      key = Random.Key(42)
    )

  private def source(embeddingSize: Int = 4) =
    Tensor(Shape(srcExtent, Axis[SrcEmb] -> embeddingSize), VType[Float32]).fill(0.5f)

  private def target = Tensor(Shape(tgtExtent, embExtent), VType[Float32]).fill(0.25f)

  describe("MultiHeadAttention"):

    it("returns the target back in its own embedding space"):
      val attention = MultiHeadFullAttention(Axis[Src], Axis[Tgt], params())
      val result = attention(source(), target)
      result.shape(Axis[Tgt]) shouldBe 3
      result.shape(Axis[TgtEmb]) shouldBe 4

    it("reduces to the output projection bias when the projection weight is zero"):
      val base = params()
      val zeroed = base.copy(outputProjection =
        AffineLayer.Params(
          weight = Tensor.like(base.outputProjection.weight).fill(0f),
          bias = Tensor.like(base.outputProjection.bias).fromArray(Array(1f, 2f, 3f, 4f))
        )
      )
      val attention = MultiHeadFullAttention(Axis[Src], Axis[Tgt], zeroed)
      val expectedRow = Tensor(Shape1(embExtent)).fromArray(Array(1f, 2f, 3f, 4f))
      val result = attention(source(), target)
      result.slice(Axis[Tgt].at(0)) should approxEqual(expectedRow, 1e-5f)
      result.slice(Axis[Tgt].at(2)) should approxEqual(expectedRow, 1e-5f)

    it("attends across a source whose embedding space differs from the target's"):
      // Head query and key extents both follow the target embedding, so the dot product lines up.
      val attention = MultiHeadFullAttention(Axis[Src], Axis[Tgt], params(Axis[SrcEmb] -> 6))
      val result = attention(source(embeddingSize = 6), target)
      result.shape(Axis[Tgt]) shouldBe 3
      result.shape(Axis[TgtEmb]) shouldBe 4

    it("restricts the first target position to the first source position under a causal mask"):
      val full = MultiHeadFullAttention(Axis[Src], Axis[Tgt], params())
      val causal = MultiHeadCausalAttention(Axis[Src], Axis[Tgt], params())
      // Rows beyond the first see different amounts of the source, so the results diverge.
      val varyingSource = Tensor(Shape(srcExtent, Axis[SrcEmb] -> 4)).fromArray(
        Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
      )
      (full(varyingSource, target) - causal(varyingSource, target)).abs.max.item should be > 1e-3f

  describe("MultiHeadAttention.applyWithIntermediates"):

    it("attends as apply does, and reports what every head worked with"):
      val attention = MultiHeadFullAttention(Axis[Src], Axis[Tgt], params())
      val (attended, intermediates) = attention.applyWithIntermediates(source(), target)
      attended should approxEqual(attention(source(), target), 1e-6f)
      intermediates.queries.shape(Axis[Head]) shouldBe numHeads
      intermediates.queries.shape(Axis[Tgt]) shouldBe 3
      intermediates.keys.shape(Axis[Src]) shouldBe 3
      intermediates.values.shape(Axis[HeadValue]) shouldBe 2

    it("reports the projections each head drew from, not the attention weights"):
      val attention = MultiHeadFullAttention(Axis[Src], Axis[Tgt], params())
      val (_, intermediates) = attention.applyWithIntermediates(source(), target)
      // The queries are the target read through each head's own query weights.
      val firstHeadQueries = target.dot(Axis[TgtEmb])(params().queryWeights.slice(Axis[Head].at(0)))
      intermediates.queries.slice(Axis[Head].at(0)) should approxEqual(firstHeadQueries, 1e-5f)

  describe("MultiHeadAttention.Params.xavierUniformDepthScaled"):

    it("splits both embedding spaces across the heads"):
      val p = params(Axis[SrcEmb] -> 6)
      p.queryWeights.shape(Axis[Head]) shouldBe numHeads
      p.queryWeights.shape(Axis[TgtEmb]) shouldBe 4
      p.queryWeights.shape(Axis[HeadQuery]) shouldBe 2
      p.keyWeights.shape(Axis[SrcEmb]) shouldBe 6
      p.keyWeights.shape(Axis[HeadKey]) shouldBe 2
      p.valueWeights.shape(Axis[HeadValue]) shouldBe 3

    it("gives the query and key spaces the same extent so the dot product contracts"):
      val p = params(Axis[SrcEmb] -> 6)
      p.queryWeights.shape(Axis[HeadQuery]) shouldBe p.keyWeights.shape(Axis[HeadKey])

    it("projects the concatenated head values back into the target embedding"):
      val p = params(Axis[SrcEmb] -> 6)
      p.outputProjection.weight.shape(Axis[Head |*| HeadValue]) shouldBe numHeads * 3
      p.outputProjection.weight.shape(Axis[TgtEmb]) shouldBe 4

    it("requires both embedding sizes to be divisible by the head count"):
      an[IllegalArgumentException] should be thrownBy
        MultiHeadAttention.Params.xavierUniformDepthScaled(2, 3, Axis[SrcEmb] -> 4, embExtent, Random.Key(42))
      an[IllegalArgumentException] should be thrownBy
        MultiHeadAttention.Params.xavierUniformDepthScaled(2, 2, Axis[SrcEmb] -> 5, embExtent, Random.Key(42))
