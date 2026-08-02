package deepwit.base

import deepwit.*
import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class LinearLayerSuite extends AnyFunSpec with Matchers:

  describe("LinearLayer"):

    it("identity"):
      val linearLayer = LinearLayer(LinearLayer.Params.identity(Axis[A] -> 5, VType[Float32]))
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      linearLayer(t) should approxEqual(t.relabelTo(Axis[Prime[A]]), 1e-6f)

    it("weight"):
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      val scalingWeight = Tensor(Shape(Axis[A] -> 5)).fromArray(Array(2f, 2f, 2f, 2f, 2f))
      val weight = Tensor2.diag(scalingWeight)
      val linearLayer = LinearLayer(LinearLayer.Params(weight = weight))
      linearLayer(t) should approxEqual((t * scalingWeight).relabelTo(Axis[Prime[A]]), 1e-6f)
