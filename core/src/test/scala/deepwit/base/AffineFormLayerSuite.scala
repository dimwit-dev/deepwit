package deepwit.base

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class AffineFormLayerSuite extends AnyFunSpec with Matchers:

  describe("AffineFormLayer"):

    it("ones weight sums the input"):
      val params = AffineFormLayer.Params(weight = Tensor(Shape1(Axis[A] -> 5)).fill(1f), bias = Tensor0(0f))
      val affineFormLayer = AffineFormLayer(params)
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      affineFormLayer(t) should approxEqual(t.sum, 1e-6f)

    it("bias"):
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      val bias = Tensor0(0.5f)
      val affineFormLayer = AffineFormLayer(AffineFormLayer.Params(weight = Tensor(Shape1(Axis[A] -> 5)).fill(1f), bias = bias))
      affineFormLayer(t) should approxEqual(t.sum + bias, 1e-6f)

    it("weight"):
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      val weight = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(2f, 0f, -1f, 0.5f, 3f))
      val bias = Tensor0(0.5f)
      val affineFormLayer = AffineFormLayer(AffineFormLayer.Params(weight = weight, bias = bias))
      affineFormLayer(t) should approxEqual((t * weight).sum + bias, 1e-6f)
      affineFormLayer(t) should approxEqual(Tensor0(16.5f), 1e-6f)

    it("xavierNormal has a zero bias and the expected shape"):
      val params = AffineFormLayer.Params.xavierNormal(Axis[A] -> 5, VType[Float32], Random.Key(42))
      params.weight.shape(Axis[A]) shouldBe 5
      params.bias should approxEqual(Tensor0(0f), 1e-6f)
