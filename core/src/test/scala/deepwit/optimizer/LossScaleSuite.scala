package deepwit.optimizer

import deepwit.*
import deepwit.base.AffineLayer
import dimwit.*
import dimwit.optimizer.Adam
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class LossScaleSuite extends AnyFunSpec with Matchers:

  private def vector(values: Float*): Tensor1[A, Float32] =
    Tensor(Shape1(Axis[A] -> values.size)).fromArray(values.toArray)

  private val params = AffineLayer.Params(
    weight = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2)).fromArray(Array(1f, 2f, 3f, 4f)),
    bias = Tensor(Shape1(Axis[B] -> 2)).fromArray(Array(5f, 6f))
  )

  private def cost(p: AffineLayer.Params[A, B, Float32]): Tensor0[Float32] =
    p.weight.pow(2).sum + p.bias.sum

  describe("unscaled"):

    it("recovers the gradients of the unscaled loss"):
      val lossScale = LossScale.initial(1024f)
      val expected = Autodiff.grad(cost)(params).value
      val grads = lossScale.unscaled(Autodiff.grad((p: AffineLayer.Params[A, B, Float32]) => lossScale.scaled(cost(p)))(params)).value
      grads.weight should approxEqual(expected.weight, 1e-5f)
      grads.bias should approxEqual(expected.bias, 1e-5f)

  describe("allFinite"):

    it("holds for finite gradients"):
      allFinite(Grad(vector(1f, -2f))).item shouldBe true

    it("fails on inf and NaN"):
      allFinite(Grad(vector(1f, Float.PositiveInfinity))).item shouldBe false
      allFinite(Grad(vector(Float.NaN, 1f))).item shouldBe false

  describe("next"):

    it("halves the scale and resets the count after an overflow"):
      val next = LossScale(Tensor0(1024f), Tensor0(7)).next(Tensor0(false))
      next.scale.item shouldBe 512f
      next.finiteSteps.item shouldBe 0

    it("counts finite steps and doubles the scale after growthInterval of them"):
      val afterTwo = (1 to 2).foldLeft(LossScale.initial(1024f))((lossScale, _) => lossScale.next(Tensor0(true), growthInterval = 3))
      afterTwo.scale.item shouldBe 1024f
      afterTwo.finiteSteps.item shouldBe 2

      val afterThree = afterTwo.next(Tensor0(true), growthInterval = 3)
      afterThree.scale.item shouldBe 2048f
      afterThree.finiteSteps.item shouldBe 0

    it("does not back off below minScale"):
      LossScale(Tensor0(1f), Tensor0(0)).next(Tensor0(false)).scale.item shouldBe 1f

  describe("select"):

    it("keeps or replaces every leaf, whatever its shape, inside jit"):
      val optimizer = Adam(Tensor0(0.1f))
      val optState = optimizer.init(params)
      val (newParams, newOptState) = optimizer.update(Autodiff.grad(cost)(params), params, optState)

      val step = jit: (finite: Tensor0[Bool]) =>
        (select(finite, newParams, params), select(finite, newOptState, optState))

      val (skippedParams, skippedOptState) = step(Tensor0(false))
      skippedParams.weight should approxEqual(params.weight, 0f)
      skippedParams.bias should approxEqual(params.bias, 0f)
      skippedOptState.momentums.weight should approxEqual(optState.momentums.weight, 0f)
      skippedOptState.beta1t.item shouldBe optState.beta1t.item

      val (takenParams, takenOptState) = step(Tensor0(true))
      takenParams.weight should approxEqual(newParams.weight, 0f)
      takenOptState.beta1t.item shouldBe newOptState.beta1t.item

  describe("in Float16"):

    val tiny = Tensor0(1e-4f).asFloat(VType[Float16])

    // The gradient is tiny · tiny = 1e-8, below Float16's smallest value of about 6e-8.
    def tinyGradientCost(w: Tensor0[Float32]): Tensor0[Float32] =
      (w.asFloat(VType[Float16]) * tiny * tiny).asFloat(VType[Float32])

    def gradWith(lossScale: LossScale): Grad[Tensor0[Float32]] =
      lossScale.unscaled(Autodiff.grad((w: Tensor0[Float32]) => lossScale.scaled(tinyGradientCost(w)))(Tensor0(1f)))

    it("loses a small gradient without scaling"):
      Autodiff.grad(tinyGradientCost)(Tensor0(1f)).value.item shouldBe 0f

    it("keeps it with scaling, after backing off from an overflow"):
      // The loss's own gradient of 1, scaled by 2^16, exceeds Float16's largest value of 65504.
      val initial = LossScale.initial()
      val overflowed = gradWith(initial)
      allFinite(overflowed).item shouldBe false

      val grad = gradWith(initial.next(allFinite(overflowed)))
      allFinite(grad).item shouldBe true
      grad.value.item shouldBe (1e-8f +- 1e-10f)
