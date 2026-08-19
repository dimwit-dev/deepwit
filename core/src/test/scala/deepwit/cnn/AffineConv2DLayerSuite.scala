package deepwit.cnn

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class AffineConv2DLayerSuite extends AnyFunSpec with Matchers:

  trait H derives Label
  trait W derives Label
  trait InChannel derives Label
  trait OutChannel derives Label

  private def unitKernel = Tensor(Shape(Axis[H] -> 1, Axis[W] -> 1, Axis[InChannel] -> 1, Axis[OutChannel] -> 1)).fromArray(Array(1f))
  private def onesKernel2x2 = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[InChannel] -> 1, Axis[OutChannel] -> 1)).fill(1f)
  private def zeroBias = Tensor(Shape1(Axis[OutChannel] -> 1)).fill(0f)

  // format: off
  private def input3x3 = Tensor(Shape(Axis[H] -> 3, Axis[W] -> 3, Axis[InChannel] -> 1)).fromArray(Array[Float](
    1f, 2f, 3f,
    4f, 5f, 6f,
    7f, 8f, 9f
  ))
  // format: on

  describe("AffineConv2DLayer"):

    it("reproduces the input for a unit kernel and a zero bias"):
      val layer = AffineConv2DLayer(AffineConv2DLayer.Params(unitKernel, zeroBias))
      // format: off
      val expected = Tensor(Shape(Axis[H] -> 3, Axis[W] -> 3, Axis[OutChannel] -> 1)).fromArray(Array[Float](
        1f, 2f, 3f,
        4f, 5f, 6f,
        7f, 8f, 9f
      ))
      // format: on
      layer(input3x3) should approxEqual(expected, 1e-6f)

    it("adds the bias to every position"):
      val bias = Tensor(Shape1(Axis[OutChannel] -> 1)).fill(5f)
      val plain = AffineConv2DLayer(AffineConv2DLayer.Params(unitKernel, zeroBias))
      val biased = AffineConv2DLayer(AffineConv2DLayer.Params(unitKernel, bias))
      biased(input3x3) should approxEqual(plain(input3x3) +! Tensor0(5f), 1e-6f)

    it("sums each window for an all-ones kernel with VALID padding"):
      val layer = AffineConv2DLayer(AffineConv2DLayer.Params(onesKernel2x2, zeroBias), padding = Padding.VALID)
      // format: off
      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[OutChannel] -> 1)).fromArray(Array[Float](
        12f, 16f,
        24f, 28f
      ))
      // format: on
      layer(input3x3) should approxEqual(expected, 1e-6f)

    it("keeps the spatial shape with SAME padding and halves it with stride two"):
      val same = AffineConv2DLayer(AffineConv2DLayer.Params(onesKernel2x2, zeroBias), padding = Padding.SAME)
      same(input3x3).shape(Axis[H]) shouldBe 3
      same(input3x3).shape(Axis[W]) shouldBe 3

      val input4x4 = Tensor(Shape(Axis[H] -> 4, Axis[W] -> 4, Axis[InChannel] -> 1), VType[Float32]).fill(1f)
      val strided = AffineConv2DLayer(AffineConv2DLayer.Params(onesKernel2x2, zeroBias), stride = 2, padding = Padding.SAME)
      strided(input4x4).shape(Axis[H]) shouldBe 2
      strided(input4x4).shape(Axis[W]) shouldBe 2

    it("maps every input channel onto every output channel"):
      // Two input channels summed into one output channel by a unit kernel.
      val kernel = Tensor(Shape(Axis[H] -> 1, Axis[W] -> 1, Axis[InChannel] -> 2, Axis[OutChannel] -> 1)).fill(1f)
      val input = Tensor(Shape(Axis[H] -> 1, Axis[W] -> 1, Axis[InChannel] -> 2)).fromArray(Array(3f, 4f))
      val layer = AffineConv2DLayer(AffineConv2DLayer.Params(kernel, zeroBias))
      layer(input) should approxEqual(Tensor(Shape(Axis[H] -> 1, Axis[W] -> 1, Axis[OutChannel] -> 1)).fill(7f), 1e-6f)

  describe("AffineConv2DLayer.Params"):

    it("xavierUniform has the kernel shape and a zero bias"):
      val params = AffineConv2DLayer.Params.xavierUniform(Axis[H] -> 3, Axis[W] -> 3, Axis[InChannel] -> 2, Axis[OutChannel] -> 4, Random.Key(42))
      params.kernel.shape(Axis[H]) shouldBe 3
      params.kernel.shape(Axis[W]) shouldBe 3
      params.kernel.shape(Axis[InChannel]) shouldBe 2
      params.kernel.shape(Axis[OutChannel]) shouldBe 4
      params.bias should approxEqual(Tensor(Shape1(Axis[OutChannel] -> 4)).fill(0f), 1e-6f)

    it("xavierUniform stays within the Glorot bounds of the flattened kernel"):
      // fanIn = 3 * 3 * 2 = 18, fanOut = 4 => a = sqrt(6 / 22)
      val params = AffineConv2DLayer.Params.xavierUniform(Axis[H] -> 3, Axis[W] -> 3, Axis[InChannel] -> 2, Axis[OutChannel] -> 4, Random.Key(42))
      params.kernel.abs.max.item should be <= math.sqrt(6.0 / 22).toFloat
