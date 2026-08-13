package deepwit.base

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class ActivationFunctionSuite extends AnyFunSpec with Matchers:

  describe("ActivationFunction"):

    it("sigmoid"):
      val t = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(0f, 20f, -20f))
      val expected = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(0.5f, 1f, 0f))
      sigmoid(t) should approxEqual(expected, 1e-5f)

    it("relu"):
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(-2f, -0.5f, 0f, 1.5f, 3f))
      val expected = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(0f, 0f, 0f, 1.5f, 3f))
      relu(t) should approxEqual(expected, 1e-6f)

    it("gelu"):
      val t = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(-1f, 0f, 1f))
      val expected = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(-0.15880796f, 0f, 0.841192f))
      gelu(t) should approxEqual(expected, 1e-4f)

    it("softmax"):
      val t = Tensor(Shape1(Axis[A] -> 4)).fromArray(Array(2f, 2f, 2f, 2f))
      val expected = Tensor(Shape1(Axis[A] -> 4)).fromArray(Array(0.25f, 0.25f, 0.25f, 0.25f))
      softmax(t) should approxEqual(expected, 1e-6f)
