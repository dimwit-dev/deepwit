package deepwit.loss

import dimwit.*
import dimwit.stats.Normal
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class RegressionSuite extends AnyFunSpec with Matchers:

  describe("SquaredError"):

    it("vanishes when the prediction hits the target"):
      SquaredError(Tensor0(2.5f), Tensor0(2.5f)).item shouldBe (0f +- 1e-6f)

    it("squares the residual"):
      SquaredError(Tensor0(1f), Tensor0(4f)).item shouldBe (9f +- 1e-5f)

    it("is symmetric in its arguments"):
      SquaredError(Tensor0(1f), Tensor0(4f)).item shouldBe
        (SquaredError(Tensor0(4f), Tensor0(1f)).item +- 1e-6f)

    it("is the negated Normal log density up to the documented constant"):
      val (target, prediction) = (Tensor0(0.7f), Tensor0(-1.2f))
      val logProb = Normal.isotropic(loc = prediction, scale = Tensor0(1f)).logProb(target).asFloat.item
      SquaredError(target, prediction).item shouldBe
        (-2f * logProb - math.log(2 * math.Pi).toFloat +- 1e-5f)

  describe("AbsoluteError"):

    it("vanishes when the prediction hits the target"):
      AbsoluteError(Tensor0(2.5f), Tensor0(2.5f)).item shouldBe (0f +- 1e-6f)

    it("takes the magnitude of the residual"):
      AbsoluteError(Tensor0(1f), Tensor0(4f)).item shouldBe (3f +- 1e-5f)

    it("grows linearly where SquaredError grows quadratically"):
      val near = AbsoluteError(Tensor0(0f), Tensor0(1f)).item
      val far = AbsoluteError(Tensor0(0f), Tensor0(10f)).item
      far shouldBe (10f * near +- 1e-4f)

  describe("Huber"):

    it("matches the halved squared error within the transition point"):
      val loss = Huber(Tensor0(0f), Tensor0(0.5f), transitionPoint = 1f)
      loss.item shouldBe (0.5f * 0.25f +- 1e-6f)

    it("grows linearly beyond the transition point"):
      val atFive = Huber(Tensor0(0f), Tensor0(5f), transitionPoint = 1f).item
      val atSix = Huber(Tensor0(0f), Tensor0(6f), transitionPoint = 1f).item
      atSix - atFive shouldBe (1f +- 1e-4f)

    it("joins its two branches continuously at the transition point"):
      val transitionPoint = 2f
      val below = Huber(Tensor0(0f), Tensor0(transitionPoint - 1e-3f), transitionPoint).item
      val above = Huber(Tensor0(0f), Tensor0(transitionPoint + 1e-3f), transitionPoint).item
      below shouldBe (above +- 1e-2f)

    it("rejects a non-positive transition point"):
      an[IllegalArgumentException] should be thrownBy Huber(Tensor0(0f), Tensor0(1f), transitionPoint = 0f)
