package deepwit.attention

import deepwit.*
import dimwit.*
import dimwit.stats.Normal
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class ReferenceMultiHeadAttentionSuite extends AnyFunSpec with Matchers:

  trait Src derives Label
  trait SrcEmb derives Label
  trait Tgt derives Label
  trait TgtEmb derives Label

  private val numHeads = 2
  private val srcExtent = Axis[Src] -> 3
  private val tgtExtent = Axis[Tgt] -> 5
  private val srcEmbExtent = Axis[SrcEmb] -> 4
  private val tgtEmbExtent = Axis[TgtEmb] -> 4

  private val batched = MultiHeadAttention.Params.xavierUniformDepthScaled(
    numTransformerLayers = 2,
    numHeads = numHeads,
    sourceEmbeddingExtent = srcEmbExtent,
    targetEmbeddingExtent = tgtEmbExtent,
    vtype = VType[Float32],
    key = Random.Key(42)
  )

  /** The weights the batched implementation holds, split head by head, so both see the same model. */
  private val perHead = ReferenceMultiHeadAttention.Params(
    heads = (0 until numHeads).toList.map(head =>
      Attention.Params(
        batched.queryWeights.slice(Axis[Head].at(head)),
        batched.keyWeights.slice(Axis[Head].at(head)),
        batched.valueWeights.slice(Axis[Head].at(head))
      )
    ),
    outputProjection = batched.outputProjection
  )

  private def source = Normal.standardNormal(Shape(srcExtent, srcEmbExtent)).sample(Random.Key(7))
  private def target = Normal.standardNormal(Shape(tgtExtent, tgtEmbExtent)).sample(Random.Key(8))

  describe("ReferenceMultiHeadAttention"):

    it("computes what MultiHeadAttention computes, under a full mask"):
      val folded = MultiHeadCustomAttention(batched, fullMask[Tgt, Src], AttentionScore.scaledDotProduct)
      val reference = ReferenceMultiHeadAttention(perHead, fullMask[Tgt, Src], AttentionScore.scaledDotProduct)
      reference(source, target) should approxEqual(folded(source, target), 1e-5f)

    it("computes what MultiHeadAttention computes, under a causal mask"):
      val folded = MultiHeadCustomAttention(batched, causalMask[Tgt, Src], AttentionScore.scaledDotProduct)
      val reference = ReferenceMultiHeadAttention(perHead, causalMask[Tgt, Src], AttentionScore.scaledDotProduct)
      reference(source, target) should approxEqual(folded(source, target), 1e-5f)
