package deepwit.cnn

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TransposeAffineConv2DLayerSuite extends AnyFunSpec with Matchers:

  trait H derives Label
  trait W derives Label
  trait InChannel derives Label
  trait OutChannel derives Label

  private def unitKernel = Tensor(Shape(Axis[H] -> 1, Axis[W] -> 1, Axis[InChannel] -> 1, Axis[OutChannel] -> 1)).fromArray(Array(1f))
  private def zeroBias = Tensor(Shape1(Axis[InChannel] -> 1)).fill(0f)

  // format: off
  private def input2x2 = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[OutChannel] -> 1)).fromArray(Array[Float](
    1f, 2f,
    3f, 4f
  ))
  // format: on

  describe("TransposeAffineConv2DLayer"):

    it("reproduces the input for a unit kernel and a zero bias"):
      val layer = TransposeAffineConv2DLayer(TransposeAffineConv2DLayer.Params(unitKernel, zeroBias))
      // format: off
      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2, Axis[InChannel] -> 1)).fromArray(Array[Float](
        1f, 2f,
        3f, 4f
      ))
      // format: on
      layer(input2x2) should approxEqual(expected, 1e-6f)

    it("adds a bias over the input channels"):
      val bias = Tensor(Shape1(Axis[InChannel] -> 1)).fill(5f)
      val plain = TransposeAffineConv2DLayer(TransposeAffineConv2DLayer.Params(unitKernel, zeroBias))
      val biased = TransposeAffineConv2DLayer(TransposeAffineConv2DLayer.Params(unitKernel, bias))
      biased(input2x2) should approxEqual(plain(input2x2) +! Tensor0(5f), 1e-6f)

    it("agrees with the linear layer given a zero bias"):
      val affine = TransposeAffineConv2DLayer(TransposeAffineConv2DLayer.Params(unitKernel, zeroBias))
      val linear = TransposeLinearConv2DLayer(TransposeLinearConv2DLayer.Params(unitKernel))
      affine(input2x2) should approxEqual(linear(input2x2), 1e-6f)

    it("upsamples the spatial shape with stride two"):
      val layer = TransposeAffineConv2DLayer(TransposeAffineConv2DLayer.Params(unitKernel, zeroBias), stride = 2, padding = Padding.SAME)
      val result = layer(input2x2)
      result.shape(Axis[H]) shouldBe 4
      result.shape(Axis[W]) shouldBe 4

  describe("TransposeAffineConv2DLayer.Params"):

    it("xavierUniform has the kernel shape and a bias over the input channels"):
      val params = TransposeAffineConv2DLayer.Params.xavierUniform(Axis[H] -> 3, Axis[W] -> 3, Axis[InChannel] -> 2, Axis[OutChannel] -> 4, Random.Key(42))
      params.kernel.shape(Axis[H]) shouldBe 3
      params.kernel.shape(Axis[W]) shouldBe 3
      params.kernel.shape(Axis[InChannel]) shouldBe 2
      params.kernel.shape(Axis[OutChannel]) shouldBe 4
      params.bias should approxEqual(Tensor(Shape1(Axis[InChannel] -> 2)).fill(0f), 1e-6f)
