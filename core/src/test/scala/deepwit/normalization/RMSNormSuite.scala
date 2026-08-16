package deepwit.normalization

import deepwit.*
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class RMSNormSuite extends AnyFunSpec with Matchers:

  private val extent = Axis[A] -> 5
  private val identityParams = RMSNorm.Params.identity(extent, VType[Float32])
  private val x = Tensor(Shape1(extent)).fromArray(Array(1f, 2f, 3f, 4f, 5f))

  describe("RMSNorm"):

    it("divides by the root mean square"):
      // rms = sqrt((1 + 4 + 9 + 16 + 25) / 5) = sqrt(11)
      val rms = math.sqrt(11.0).toFloat
      val expected = Tensor(Shape1(extent)).fromArray(Array(1f / rms, 2f / rms, 3f / rms, 4f / rms, 5f / rms))
      RMSNorm(identityParams)(x) should approxEqual(expected, 1e-5f)

    it("normalizes to unit root mean square"):
      val y = RMSNorm(identityParams)(x)
      y.pow(2).mean.sqrt.item shouldBe (1f +- 1e-4f)

    it("does not re-center, unlike LayerNorm"):
      // A constant input already has unit RMS after scaling, so it is preserved rather than zeroed.
      val constantExtent = Axis[A] -> 3
      val constant = Tensor(Shape1(constantExtent)).fill(2f)
      val y = RMSNorm(RMSNorm.Params.identity(constantExtent, VType[Float32]))(constant)
      y should approxEqual(Tensor(Shape1(constantExtent)).fill(1f), 1e-5f)
      y.mean.item should be > 0.5f

    it("is invariant to positive scaling but not to shifting"):
      val rmsNorm = RMSNorm(identityParams)
      rmsNorm(x *! Tensor0(7f)) should approxEqual(rmsNorm(x), 1e-4f)
      (rmsNorm(x +! Tensor0(100f)) - rmsNorm(x)).abs.max.item should be > 0.1f

    it("applies the weight after rescaling"):
      val weight = Tensor(Shape1(extent)).fromArray(Array(2f, 2f, 2f, 2f, 2f))
      val rescaled = RMSNorm(identityParams)(x)
      RMSNorm(RMSNorm.Params(weight))(x) should approxEqual(rescaled * weight, 1e-5f)

    it("maps a zero input to zeros without producing NaN"):
      val zeros = Tensor(Shape1(extent)).fill(0f)
      RMSNorm(identityParams)(zeros) should approxEqual(zeros, 1e-5f)
