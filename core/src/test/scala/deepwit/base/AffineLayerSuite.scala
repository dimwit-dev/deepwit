package deepwit.base

import deepwit.*
import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class AffineLayerSuite extends AnyFunSpec with Matchers:

  describe("AffineLayer"):

    it("identity"):
      val affineLayer = AffineLayer(AffineLayer.Params.identity(Axis[A] -> 5))
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      affineLayer(t) should approxEqual(t.relabelTo(Axis[Prime[A]]), 1e-6f)

    it("bias"):
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      val bias = Tensor(Shape(Axis[Prime[A]] -> 5)).fromArray(Array(0.5f, 1f, 1.5f, 2f, 2.5f))
      val affineLayer = AffineLayer(AffineLayer.Params.identity(Axis[A] -> 5).copy(bias = bias))
      affineLayer(t) should approxEqual(t.relabelTo(Axis[Prime[A]]) + bias, 1e-6f)

    it("weight"):
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      val scalingWeight = Tensor(Shape(Axis[A] -> 5)).fromArray(Array(2f, 2f, 2f, 2f, 2f))
      val weight = Tensor2.diag(scalingWeight)
      val affineLayer = AffineLayer(AffineLayer.Params.identity(Axis[A] -> 5).copy(weight = weight))
      affineLayer(t) should approxEqual((t * scalingWeight).relabelTo(Axis[Prime[A]]), 1e-6f)
