package deepwit.normalization

import deepwit.*
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class LayerNormSuite extends AnyFunSpec with Matchers:

  private val extent = Axis[A] -> 5
  private val identityParams = LayerNorm.Params.identity(extent, VType[Float32])
  private val x = Tensor(Shape1(extent)).fromArray(Array(1f, 2f, 3f, 4f, 5f))

  describe("LayerNorm"):

    it("standardizes to zero mean and unit variance"):
      val y = LayerNorm(identityParams)(x)
      y.mean.item shouldBe (0f +- 1e-5f)
      (y -! y.mean).pow(2).mean.item shouldBe (1f +- 1e-4f)

    it("matches the hand-computed standardization"):
      // mean = 3, population variance = 2, std = sqrt(2)
      val s = math.sqrt(2.0).toFloat
      val expected = Tensor(Shape1(extent)).fromArray(Array(-2f / s, -1f / s, 0f, 1f / s, 2f / s))
      LayerNorm(identityParams)(x) should approxEqual(expected, 1e-5f)

    it("applies weight and bias after standardizing"):
      val weight = Tensor(Shape1(extent)).fromArray(Array(2f, 2f, 2f, 2f, 2f))
      val bias = Tensor(Shape1(extent)).fromArray(Array(1f, 1f, 1f, 1f, 1f))
      val standardized = LayerNorm(identityParams)(x)
      LayerNorm(LayerNorm.Params(weight, bias))(x) should approxEqual(standardized * weight + bias, 1e-5f)

    it("is invariant to shifting and positive scaling of the input"):
      val layerNorm = LayerNorm(identityParams)
      layerNorm(x +! Tensor0(100f)) should approxEqual(layerNorm(x), 1e-4f)
      layerNorm(x *! Tensor0(7f)) should approxEqual(layerNorm(x), 1e-4f)

    it("maps a constant input to zeros without producing NaN"):
      val constant = Tensor(Shape1(extent)).fill(3f)
      val y = LayerNorm(identityParams)(constant)
      y should approxEqual(Tensor(Shape1(extent)).fill(0f), 1e-5f)
