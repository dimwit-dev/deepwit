package deepwit.cnn

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class LinearConv2DLayerSuite extends AnyFunSpec with Matchers:

  trait H derives Label
  trait W derives Label
  trait InChannel derives Label
  trait OutChannel derives Label

  private def unitKernel = Tensor(Shape(Axis[H] -> 1, Axis[W] -> 1, Axis[InChannel] -> 1, Axis[OutChannel] -> 1)).fromArray(Array(1f))
  private def onesKernel2x2 = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[InChannel] -> 1, Axis[OutChannel] -> 1)).fill(1f)

  // format: off
  private def input3x3 = Tensor(Shape(Axis[H] -> 3, Axis[W] -> 3, Axis[InChannel] -> 1)).fromArray(Array[Float](
    1f, 2f, 3f,
    4f, 5f, 6f,
    7f, 8f, 9f
  ))
  // format: on

  describe("LinearConv2DLayer"):

    it("reproduces the input for a unit kernel"):
      val layer = LinearConv2DLayer(LinearConv2DLayer.Params(unitKernel))
      // format: off
      val expected = Tensor(Shape(Axis[H] -> 3, Axis[W] -> 3, Axis[OutChannel] -> 1)).fromArray(Array[Float](
        1f, 2f, 3f,
        4f, 5f, 6f,
        7f, 8f, 9f
      ))
      // format: on
      layer(input3x3) should approxEqual(expected, 1e-6f)

    it("sums each window for an all-ones kernel with VALID padding"):
      val layer = LinearConv2DLayer(LinearConv2DLayer.Params(onesKernel2x2), padding = Padding.VALID)
      // format: off
      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[OutChannel] -> 1)).fromArray(Array[Float](
        12f, 16f,
        24f, 28f
      ))
      // format: on
      layer(input3x3) should approxEqual(expected, 1e-6f)

    it("agrees with the affine layer given a zero bias"):
      val kernel = onesKernel2x2
      val linear = LinearConv2DLayer(LinearConv2DLayer.Params(kernel))
      val affine = AffineConv2DLayer(AffineConv2DLayer.Params(kernel, Tensor(Shape1(Axis[OutChannel] -> 1)).fill(0f)))
      linear(input3x3) should approxEqual(affine(input3x3), 1e-6f)

    it("halves the spatial shape with stride two"):
      val input4x4 = Tensor(Shape(Axis[H] -> 4, Axis[W] -> 4, Axis[InChannel] -> 1), VType[Float32]).fill(1f)
      val strided = LinearConv2DLayer(LinearConv2DLayer.Params(onesKernel2x2), stride = 2, padding = Padding.SAME)
      strided(input4x4).shape(Axis[H]) shouldBe 2
      strided(input4x4).shape(Axis[W]) shouldBe 2

  describe("LinearConv2DLayer.Params"):

    it("xavierUniform has the kernel shape"):
      val params = LinearConv2DLayer.Params.xavierUniform(Axis[H] -> 3, Axis[W] -> 3, Axis[InChannel] -> 2, Axis[OutChannel] -> 4, VType[Float32], Random.Key(42))
      params.kernel.shape(Axis[H]) shouldBe 3
      params.kernel.shape(Axis[W]) shouldBe 3
      params.kernel.shape(Axis[InChannel]) shouldBe 2
      params.kernel.shape(Axis[OutChannel]) shouldBe 4
