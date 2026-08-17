package deepwit.attention

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class MaskingSuite extends AnyFunSpec with Matchers:

  describe("causalMask"):

    it("is lower triangular including the diagonal"):
      val mask = causalMask(Shape2(Axis[A] -> 3, Axis[B] -> 3))
      // format: off
      val expected = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 3)).fromArray(Array[Float](
        1f, 0f, 0f,
        1f, 1f, 0f,
        1f, 1f, 1f
      ))
      // format: on
      mask.asFloat32 should approxEqual(expected, 1e-6f)

    it("lets row i attend to exactly i + 1 positions"):
      val mask = causalMask(Shape2(Axis[A] -> 4, Axis[B] -> 4))
      val visible = mask.asFloat32.sum(Axis[B])
      visible should approxEqual(Tensor(Shape1(Axis[A] -> 4)).fromArray(Array(1f, 2f, 3f, 4f)), 1e-6f)

    it("works on a non-square score shape"):
      val mask = causalMask(Shape2(Axis[A] -> 2, Axis[B] -> 4))
      mask.shape(Axis[A]) shouldBe 2
      mask.shape(Axis[B]) shouldBe 4
      val visible = mask.asFloat32.sum(Axis[B])
      visible should approxEqual(Tensor(Shape1(Axis[A] -> 2)).fromArray(Array(1f, 2f)), 1e-6f)

  describe("fullMask"):

    it("lets every position attend everywhere"):
      val mask = fullMask(Shape2(Axis[A] -> 2, Axis[B] -> 3))
      mask.asFloat32 should approxEqual(Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(1f), 1e-6f)
