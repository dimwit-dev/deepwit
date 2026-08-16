package deepwit.cnn

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TransposeLinearConv2DLayerSuite extends AnyFunSpec with Matchers:

  trait H derives Label
  trait W derives Label
  trait InChannel derives Label
  trait OutChannel derives Label

  private def unitKernel = Tensor(Shape(Axis[H] -> 1, Axis[W] -> 1, Axis[InChannel] -> 1, Axis[OutChannel] -> 1)).fromArray(Array(1f))

  // format: off
  private def input2x2 = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[OutChannel] -> 1)).fromArray(Array[Float](
    1f, 2f,
    3f, 4f
  ))
  // format: on

  describe("TransposeLinearConv2DLayer"):

    it("reproduces the input for a unit kernel"):
      val layer = TransposeLinearConv2DLayer(TransposeLinearConv2DLayer.Params(unitKernel))
      // format: off
      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[InChannel] -> 1)).fromArray(Array[Float](
        1f, 2f,
        3f, 4f
      ))
      // format: on
      layer(input2x2) should approxEqual(expected, 1e-6f)

    it("maps the output channels back onto the input channels"):
      val layer = TransposeLinearConv2DLayer(TransposeLinearConv2DLayer.Params(unitKernel))
      val result = layer(input2x2)
      result.shape(Axis[InChannel]) shouldBe 1

    it("upsamples the spatial shape with stride two"):
      val layer = TransposeLinearConv2DLayer(TransposeLinearConv2DLayer.Params(unitKernel), stride = 2, padding = Padding.SAME)
      val result = layer(input2x2)
      result.shape(Axis[H]) shouldBe 4
      result.shape(Axis[W]) shouldBe 4

  describe("TransposeLinearConv2DLayer.Params"):

    it("xavierUniform has the kernel shape"):
      val params = TransposeLinearConv2DLayer.Params.xavierUniform(Axis[H] -> 3, Axis[W] -> 3, Axis[InChannel] -> 2, Axis[OutChannel] -> 4, VType[Float32], Random.Key(42))
      params.kernel.shape(Axis[H]) shouldBe 3
      params.kernel.shape(Axis[W]) shouldBe 3
      params.kernel.shape(Axis[InChannel]) shouldBe 2
      params.kernel.shape(Axis[OutChannel]) shouldBe 4
