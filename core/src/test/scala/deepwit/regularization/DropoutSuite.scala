package deepwit.regularization

import deepwit.*
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class DropoutSuite extends AnyFunSpec with Matchers:

  trait Feature derives Label

  private val featureExtent = Axis[Feature] -> 4
  private val identityParams = Dropout.Params.identity(featureExtent, VType[Float32])

  private val features = Tensor(Shape1(featureExtent)).fromArray(Array(1f, 2f, 3f, 4f))

  /** The features a thinned projection lets through, as a mask of ones and zeros. */
  private def survivors(params: Dropout.Params[Feature, Float32]): Seq[Float] =
    val passed = Dropout(params)(Tensor(Shape1(featureExtent)).fill(1f))
    (0 until featureExtent.size).map(i => if passed.slice(Axis[Feature].at(i)).item == 0f then 0f else 1f)

  describe("Dropout"):

    it("is the identity while its projection is untouched"):
      Dropout(identityParams)(features) should approxEqual(features, 1e-6f)

    it("deletes the features its thinned projection drops and rescales those that remain"):
      val thinned = identityParams.thinned(probability = 0.5f, key = Random.Key(42))
      val kept = survivors(thinned)
      kept should contain(0f) // this draw deleted something, so there is a deletion to look at
      val expected = (0 until featureExtent.size).map: i =>
        val x = features.slice(Axis[Feature].at(i)).item
        if kept(i) == 0f then 0f else x * 2f // survivors are scaled by 1 / (1 - 0.5)
      Dropout(thinned)(features) should approxEqual(Tensor(Shape1(featureExtent)).fromArray(expected.toArray), 1e-5f)

    it("sends no gradient into a deleted feature, and the survivor scale into the rest"):
      // What a training step does: thin the parameters, then differentiate the model that uses them.
      val thinned = identityParams.thinned(probability = 0.5f, key = Random.Key(42))
      def cost(input: Tensor1[Feature, Float32]): Tensor0[Float32] = Dropout(thinned)(input).sum
      val gradient = Autodiff.grad(cost)(features).value
      survivors(thinned).zipWithIndex.foreach: (survived, i) =>
        val g = gradient.slice(Axis[Feature].at(i)).item
        if survived == 0f then g shouldBe 0f else g shouldBe 2f +- 1e-5f
