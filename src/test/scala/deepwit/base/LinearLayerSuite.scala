package deepwit.base

import deepwit.approxEqual
import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

trait A derives Label

class LinearLayerSuite extends AnyFunSpec with Matchers:

  describe("LinearLayer"):

    it("identity"):
      val linearEyeParams = LinearLayer.Params(weight = Tensor2.eye(Axis[A] -> 5))
      val linearLayer = LinearLayer(linearEyeParams)
      val t = Tensor(Shape1(Axis[A] -> 5)).fromArray(Array(1f, 2f, 3f, 4f, 5f))
      linearLayer(t) should approxEqual(t.relabelTo(Axis[Prime[A]]), 1e-6f)
