package deepwit.attention

import deepwit.*
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class AttentionSuite extends AnyFunSpec with Matchers:

  trait Src derives Label
  trait SrcEmb derives Label
  trait Tgt derives Label
  trait TgtEmb derives Label
  trait Qry derives Label
  trait Ky derives Label
  trait Vl derives Label

  private val srcExtent = Axis[Src] -> 2
  private val tgtExtent = Axis[Tgt] -> 2

  /** Two source positions with two-dimensional embeddings. */
  // format: off
  private val source = Tensor(Shape(srcExtent, Axis[SrcEmb] -> 2)).fromArray(Array[Float](
    1f, 2f,
    3f, 4f
  ))

  private val target = Tensor(Shape(tgtExtent, Axis[TgtEmb] -> 2)).fromArray(Array[Float](
    1f, 0f,
    0f, 1f
  ))
  // format: on

  private def identity2[L1: Label, L2: Label](l1: Axis[L1], l2: Axis[L2]) =
    Tensor(Shape(l1 -> 2, l2 -> 2)).fromArray(Array(1f, 0f, 0f, 1f))

  private def zeros2[L1: Label, L2: Label](l1: Axis[L1], l2: Axis[L2]) =
    Tensor(Shape(l1 -> 2, l2 -> 2), VType[Float32]).fill(0f)

  /** Queries pass the target through, values pass the source through, keys are configurable. */
  private def paramsWithKeyWeights(wk: Tensor2[SrcEmb, Ky, Float32]) =
    Attention.Params(
      wq = identity2(Axis[TgtEmb], Axis[Qry]),
      wk = wk,
      wv = identity2(Axis[SrcEmb], Axis[Vl])
    )

  describe("ScaledDotProduct"):

    it("scales the dot products by the square root of the key dimension"):
      // format: off
      val queries = Tensor(Shape(tgtExtent, Axis[Qry] -> 2)).fromArray(Array[Float](
        1f, 0f,
        0f, 1f
      ))
      val keys = Tensor(Shape(srcExtent, Axis[Ky] -> 2)).fromArray(Array[Float](
        1f, 0f,
        1f, 1f
      ))
      val expected = Tensor(Shape(tgtExtent, srcExtent)).fromArray(Array[Float](
        1f, 1f,
        0f, 1f
      )) /! math.sqrt(2.0)
      // format: on
      ScaledDotProduct[Tgt, Src, Qry, Ky, Float32]()(queries, keys) should approxEqual(expected, 1e-6f)

  describe("Attention"):

    it("returns one value vector per target position"):
      val attention = FullAttention(Axis[Src], Axis[Tgt], paramsWithKeyWeights(identity2(Axis[SrcEmb], Axis[Ky])))
      val result = attention(source, target)
      result.shape(Axis[Tgt]) shouldBe 2
      result.shape(Axis[Vl]) shouldBe 2

    it("averages the values when every key is identical"):
      // Zero key weights make all scores equal, so softmax spreads the attention uniformly.
      val attention = FullAttention(Axis[Src], Axis[Tgt], paramsWithKeyWeights(zeros2(Axis[SrcEmb], Axis[Ky])))
      val meanOfValues = Tensor(Shape(tgtExtent, Axis[Vl] -> 2)).fromArray(Array(2f, 3f, 2f, 3f))
      attention(source, target) should approxEqual(meanOfValues, 1e-5f)

    it("gives the first target position only the first source value under a causal mask"):
      val attention = CausalAttention(Axis[Src], Axis[Tgt], paramsWithKeyWeights(identity2(Axis[SrcEmb], Axis[Ky])))
      val firstRow = attention(source, target).slice(Axis[Tgt].at(0))
      firstRow should approxEqual(Tensor(Shape1(Axis[Vl] -> 2)).fromArray(Array(1f, 2f)), 1e-5f)

    it("honours a custom attention score function"):
      // Constant scores make the attention uniform regardless of the key weights.
      def constantScores(queries: Tensor2[Tgt, Qry, Float32], keys: Tensor2[Src, Ky, Float32]): Tensor2[Tgt, Src, Float32] =
        Tensor(Shape(queries.shape.extent(Axis[Tgt]), keys.shape.extent(Axis[Src])), VType[Float32]).fill(0f)

      val attention = FullAttention(Axis[Src], Axis[Tgt], paramsWithKeyWeights(identity2(Axis[SrcEmb], Axis[Ky])), constantScores)
      val meanOfValues = Tensor(Shape(tgtExtent, Axis[Vl] -> 2)).fromArray(Array(2f, 3f, 2f, 3f))
      attention(source, target) should approxEqual(meanOfValues, 1e-5f)

    it("differs from the uniform result under the default scores"):
      val default = FullAttention(Axis[Src], Axis[Tgt], paramsWithKeyWeights(identity2(Axis[SrcEmb], Axis[Ky])))
      val uniform = FullAttention(Axis[Src], Axis[Tgt], paramsWithKeyWeights(zeros2(Axis[SrcEmb], Axis[Ky])))
      (default(source, target) - uniform(source, target)).abs.max.item should be > 1e-3f

  describe("Attention.Params.init"):

    it("shapes the projections from the embeddings into the query, key and value spaces"):
      val params = Attention.Params.init(
        queryExtent = Axis[Qry] -> 3,
        keyExtent = Axis[Ky] -> 3,
        valueExtent = Axis[Vl] -> 5,
        sourceEmbeddingExtent = Axis[SrcEmb] -> 6,
        targetEmbeddingExtent = Axis[TgtEmb] -> 4,
        vtype = VType[Float32],
        key = Random.Key(42)
      )
      params.queryWeights.weight.shape(Axis[TgtEmb]) shouldBe 4
      params.queryWeights.weight.shape(Axis[Qry]) shouldBe 3
      params.keyWeights.weight.shape(Axis[SrcEmb]) shouldBe 6
      params.keyWeights.weight.shape(Axis[Ky]) shouldBe 3
      params.valueWeights.weight.shape(Axis[SrcEmb]) shouldBe 6
      params.valueWeights.weight.shape(Axis[Vl]) shouldBe 5
