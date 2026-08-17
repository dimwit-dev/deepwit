package deepwit.transformer

import deepwit.*
import deepwit.normalization.LayerNorm
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class CrossTransformerSuite extends AnyFunSpec with Matchers:

  trait CrossCtx derives Label
  trait CrossEmb derives Label
  trait Ctx derives Label
  trait Emb derives Label

  private val numLayers = 3
  private val crossCtxExtent = Axis[CrossCtx] -> 3
  private val ctxExtent = Axis[Ctx] -> 4
  private val embExtent = Axis[Emb] -> 4

  private def params(crossEmbeddingExtent: AxisExtent[CrossEmb] = Axis[CrossEmb] -> 4) =
    CrossTransformer.Params.xavierUniformDepthScaled(
      numTransformerLayers = numLayers,
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

  describe("CrossTransformer"):

    it("preserves the shape of the context"):
      val transformer = CrossTransformer(Axis[CrossCtx], Axis[Ctx], params())
      val result = transformer(crossContext(), context(13f))
      result.shape(Axis[Ctx]) shouldBe 4
      result.shape(Axis[Emb]) shouldBe 4

    it("is just the final normalization when it has no layers"):
      val normParams = LayerNorm.Params.identity(embExtent, VType[Float32])
      val transformer = CrossTransformer(Axis[CrossCtx], Axis[Ctx], CrossTransformer.Params[CrossEmb, Emb, Float32](List.empty, normParams))
      val x = context(13f)
      transformer(crossContext(), x) should approxEqual(x.vmap(Axis[Ctx])(LayerNorm(normParams)), 1e-5f)

    it("attends onto a cross context whose embedding space differs from its own"):
      val transformer = CrossTransformer(Axis[CrossCtx], Axis[Ctx], params(Axis[CrossEmb] -> 6))
      val result = transformer(crossContext(embeddingSize = 6), context(13f))
      result.shape(Axis[Ctx]) shouldBe 4
      result.shape(Axis[Emb]) shouldBe 4

    it("lets every position see every other one, self-attending in full"):
      val transformer = CrossTransformer(Axis[CrossCtx], Axis[Ctx], params())
      val a = transformer(crossContext(), context(13f))
      val b = transformer(crossContext(), context(-99f))
      (a.slice(Axis[Ctx].at(0 until 3)) - b.slice(Axis[Ctx].at(0 until 3))).abs.max.item should be > 1e-3f

  describe("CrossTransformer.applyWithHiddenStates"):

    it("returns one hidden state per layer"):
      val transformer = CrossTransformer(Axis[CrossCtx], Axis[Ctx], params())
      val (hiddenStates, _) = transformer.applyWithHiddenStates(crossContext(), context(13f))
      hiddenStates.size shouldBe numLayers

    it("normalizes its last hidden state into the same result as apply"):
      val transformer = CrossTransformer(Axis[CrossCtx], Axis[Ctx], params())
      val cross = crossContext()
      val x = context(13f)
      val (hiddenStates, result) = transformer.applyWithHiddenStates(cross, x)
      result should approxEqual(transformer(cross, x), 1e-5f)
      val normalized = hiddenStates.last.vmap(Axis[Ctx])(LayerNorm(params().finalNorm))
      result should approxEqual(normalized, 1e-5f)

    it("does not include the initial context among the hidden states"):
      val transformer = CrossTransformer(Axis[CrossCtx], Axis[Ctx], params())
      val x = context(13f)
      val (hiddenStates, _) = transformer.applyWithHiddenStates(crossContext(), x)
      (hiddenStates.head - x).abs.max.item should be > 1e-3f

  describe("CrossTransformer.Params.xavierUniformDepthScaled"):

    it("builds one parameter set per layer plus the final normalization"):
      val p = params()
      p.transformerLayers.size shouldBe numLayers
      p.finalNorm.weight should approxEqual(Tensor(Shape1(embExtent)).fill(1f), 1e-6f)
