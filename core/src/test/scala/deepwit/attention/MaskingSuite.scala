package deepwit.attention

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class MaskingSuite extends AnyFunSpec with Matchers:

  private def positionsA(range: Range) = Tensor1(Axis[A]).fromRange(range)
  private def positionsB(range: Range) = Tensor1(Axis[B]).fromRange(range)

  describe("causalMask"):

    it("is lower triangular including the diagonal"):
      val mask = causalMask(positionsA(0 until 3), positionsB(0 until 3))
      // format: off
      val expected = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 3)).fromArray(Array[Float](
        1f, 0f, 0f,
        1f, 1f, 0f,
        1f, 1f, 1f
      ))
      // format: on
      mask.asFloat32 should approxEqual(expected, 1e-6f)

    it("lets row i attend to exactly i + 1 positions"):
      val mask = causalMask(positionsA(0 until 4), positionsB(0 until 4))
      val visible = mask.asFloat32.sum(Axis[B])
      visible should approxEqual(Tensor(Shape1(Axis[A] -> 4)).fromArray(Array(1f, 2f, 3f, 4f)), 1e-6f)

    it("works on a non-square score shape"):
      val mask = causalMask(positionsA(0 until 2), positionsB(0 until 4))
      mask.shape(Axis[A]) shouldBe 2
      mask.shape(Axis[B]) shouldBe 4
      val visible = mask.asFloat32.sum(Axis[B])
      visible should approxEqual(Tensor(Shape1(Axis[A] -> 2)).fromArray(Array(1f, 2f)), 1e-6f)

    it("lets a single position attend to itself and every position before it, as when decoding"):
      val mask = causalMask(positionsA(2 to 2), positionsB(0 until 4))
      mask.asFloat32 should approxEqual(Tensor(Shape(Axis[A] -> 1, Axis[B] -> 4)).fromArray(Array(1f, 1f, 1f, 0f)), 1e-6f)

  describe("fullMask"):

    it("lets every position attend everywhere"):
      val mask = fullMask(positionsA(0 until 2), positionsB(0 until 3))
      mask.asFloat32 should approxEqual(Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(1f), 1e-6f)
