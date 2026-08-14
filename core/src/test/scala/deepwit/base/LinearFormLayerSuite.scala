package deepwit.base

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class LinearFormLayerSuite extends AnyFunSpec with Matchers:

  describe("LinearFormLayer"):

    it("ones weight sums the input"):
      val linearFormLayer = LinearFormLayer(Tensor(Shape1(Axis[A] -> 5)).fill(1f))
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      linearFormLayer(t) should approxEqual(t.sum, 1e-6f)

    it("weight"):
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      val weight = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(2f, 0f, -1f, 0.5f, 3f))
      val linearFormLayer = LinearFormLayer(LinearFormLayer.Params(weight = weight))
      linearFormLayer(t) should approxEqual((t * weight).sum, 1e-6f)
      linearFormLayer(t) should approxEqual(Tensor0(16f), 1e-6f)

    it("linearity"):
      val weight = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(2f, 0f, -1f, 0.5f, 3f))
      val linearFormLayer = LinearFormLayer(weight)
      val x = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      val y = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(-1f, 0f, 2f, 1f, 0.5f))
      linearFormLayer((x *! Tensor0(3f)) + y) should approxEqual(linearFormLayer(x) * Tensor0(3f) + linearFormLayer(y), 1e-5f)

    it("xavierUniform stays within the Glorot bounds"):
      // fanIn = 5, fanOut = 1 => a = sqrt(3 * 2 / 6) = 1
      val params = LinearFormLayer.Params.xavierUniform(Axis[A] -> 5, VType[Float32], Random.Key(42))
      params.weight.shape(Axis[A]) shouldBe 5
      params.weight.abs.max.item should be <= 1f
