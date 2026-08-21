package deepwit.loss

import deepwit.*
import deepwit.activation.{sigmoid, softmax}
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class CrossEntropySuite extends AnyFunSpec with Matchers:

  private def logits(values: Float*) = Tensor(Shape1(Axis[A] -> values.size)).fromArray(values.toArray)

  describe("CategoricalCrossEntropy.fromLogits"):

    it("costs log(C) when all classes are equally likely"):
      val loss = CategoricalCrossEntropy.fromLogits(Tensor0(2), logits(0f, 0f, 0f, 0f))
      loss.item shouldBe (math.log(4).toFloat +- 1e-5f)

    it("is invariant to a constant shift of all logits"):
      val raw = logits(1f, -2f, 0.5f, 3f)
      val shifted = raw +! Tensor0(10f)
      CategoricalCrossEntropy.fromLogits(Tensor0(1), shifted).item shouldBe
        (CategoricalCrossEntropy.fromLogits(Tensor0(1), raw).item +- 1e-5f)

    it("agrees with the probability form"):
      val raw = logits(1f, -2f, 0.5f, 3f)
      CategoricalCrossEntropy.fromLogits(Tensor0(3), raw).item shouldBe
        (CategoricalCrossEntropy(Tensor0(3), softmax(raw)).item +- 1e-5f)

    it("approaches zero for a confidently correct prediction"):
      val loss = CategoricalCrossEntropy.fromLogits(Tensor0(0), logits(100f, 0f))
      loss.item shouldBe (0f +- 1e-5f)

    it("stays finite for a confidently wrong prediction"):
      val loss = CategoricalCrossEntropy.fromLogits(Tensor0(0), logits(0f, 100f))
      loss.item shouldBe (100f +- 1e-3f)

  describe("CategoricalCrossEntropy.stable"):

    it("clips a zero probability instead of diverging"):
      val prediction = Tensor(Shape1(Axis[A] -> 2)).fromArray(Array(0f, 1f))
      val loss = CategoricalCrossEntropy.stable(ε = Tensor0(1e-7f))(Tensor0(0), prediction)
      loss.item shouldBe (-math.log(1e-7).toFloat +- 1e-2f)

  describe("BinaryCrossEntropy.fromLogits"):

    it("costs log(2) at a logit of zero"):
      BinaryCrossEntropy.fromLogits(Tensor0(1f), Tensor0(0f)).item shouldBe (math.log(2).toFloat +- 1e-6f)

    it("agrees with the probability form"):
      val logit = Tensor0(1.3f)
      BinaryCrossEntropy.fromLogits(Tensor0(1f), logit).item shouldBe
        (BinaryCrossEntropy(Tensor0(1f), sigmoid(logit)).item +- 1e-5f)

    it("stays finite for saturating logits"):
      BinaryCrossEntropy.fromLogits(Tensor0(1f), Tensor0(100f)).item shouldBe (0f +- 1e-5f)
      BinaryCrossEntropy.fromLogits(Tensor0(1f), Tensor0(-100f)).item shouldBe (100f +- 1e-3f)

    it("is symmetric in the target"):
      BinaryCrossEntropy.fromLogits(Tensor0(0f), Tensor0(2f)).item shouldBe
        (BinaryCrossEntropy.fromLogits(Tensor0(1f), Tensor0(-2f)).item +- 1e-6f)

  describe("BernoulliCrossEntropy"):

    it("treats true as one and false as zero"):
      val prediction = Tensor0(0.3f)
      BernoulliCrossEntropy(Tensor0(true), prediction).item shouldBe
        (BinaryCrossEntropy(Tensor0(1f), prediction).item +- 1e-6f)
      BernoulliCrossEntropy(Tensor0(false), prediction).item shouldBe
        (BinaryCrossEntropy(Tensor0(0f), prediction).item +- 1e-6f)

    it("agrees with its own logit form"):
      val logit = Tensor0(-0.7f)
      BernoulliCrossEntropy.fromLogits(Tensor0(true), logit).item shouldBe
        (BernoulliCrossEntropy(Tensor0(true), sigmoid(logit)).item +- 1e-5f)
