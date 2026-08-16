package deepwit.optimizer

import deepwit.*
import deepwit.base.AffineLayer
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class GradientClippingSuite extends AnyFunSpec with Matchers:

  private def vector(values: Float*): Tensor1[A, Float32] =
    Tensor(Shape1(Axis[A] -> values.size)).fromArray(values.toArray)

  private def clip(g: Tensor1[A, Float32], maxNorm: Float): Tensor1[A, Float32] =
    Grad(g).clipGlobalNorm(Tensor0(maxNorm)).value

  describe("clipGlobalNorm"):

    it("leaves gradients below the threshold untouched"):
      // norm of (3, 4) is 5
      val g = vector(3f, 4f)
      clip(g, 10f) should approxEqual(g, 1e-6f)

    it("scales gradients above the threshold down to the threshold"):
      val clipped = clip(vector(3f, 4f), 1f)
      clipped.pow(2).sum.sqrt.item shouldBe (1f +- 1e-4f)

    it("preserves the direction of the gradients"):
      clip(vector(3f, 4f), 1f) should approxEqual(vector(0.6f, 0.8f), 1e-4f)

    it("measures the norm across every leaf of the tree"):
      // Leaves of norm 3 and 4 give a global norm of 5, so clipping to 5 is a no-op.
      val tree = AffineLayer.Params(
        weight = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2)).fromArray(Array(3f, 0f, 0f, 0f)),
        bias = Tensor(Shape1(Axis[B] -> 2)).fromArray(Array(4f, 0f))
      )

      val unchanged: AffineLayer.Params[A, B, Float32] = Grad(tree).clipGlobalNorm(Tensor0(5f)).value
      unchanged.weight should approxEqual(tree.weight, 1e-4f)
      unchanged.bias should approxEqual(tree.bias, 1e-4f)

      val halved: AffineLayer.Params[A, B, Float32] = Grad(tree).clipGlobalNorm(Tensor0(2.5f)).value
      halved.weight should approxEqual(tree.weight *! Tensor0(0.5f), 1e-4f)
      halved.bias should approxEqual(tree.bias *! Tensor0(0.5f), 1e-4f)
