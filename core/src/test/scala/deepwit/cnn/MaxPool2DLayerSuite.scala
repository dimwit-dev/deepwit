package deepwit.cnn

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class MaxPool2DLayerSuite extends AnyFunSpec with Matchers:

  trait H derives Label
  trait W derives Label

  describe("MaxPool2DLayer"):

    it("should compute 2x2 max pooling with stride 2 (downsampling)"):
      val layer = MaxPool2DLayer[H, W, Float32](window = 2, stride = 2, padding = Padding.VALID)

      // format: off
      val input = Tensor(Shape(Axis[H] -> 4, Axis[W] -> 4)).fromArray(Array[Float](
         1f,  2f,  3f,  4f,
         5f,  6f,  7f,  8f,
         9f, 10f, 11f, 12f,
        13f, 14f, 15f, 16f
      ))

      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2)).fromArray(Array[Float](
         6f,  8f,
        14f, 16f
      ))
      // format: on

      val result = layer(input)
      result should approxEqual(expected, 1e-6f)

    it("should compute 2x2 max pooling with stride 1 (overlapping windows)"):
      val layer = MaxPool2DLayer[H, W, Float32](window = 2, stride = 1, padding = Padding.VALID)

      // format: off
      val input = Tensor(Shape(Axis[H] -> 3, Axis[W] -> 3)).fromArray(Array[Float](
        1f, 2f, 3f,
        4f, 5f, 6f,
        7f, 8f, 9f
      ))

      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2)).fromArray(Array[Float](
        5f, 6f,
        8f, 9f
      ))
      // format: on

      val result = layer(input)
      result should approxEqual(expected, 1e-6f)

    it("should correctly handle SAME padding"):
      val layer = MaxPool2DLayer[H, W, Float32](window = 2, stride = 1, padding = Padding.SAME)

      // format: off
      val input = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2)).fromArray(Array[Float](
        1f, 2f,
        3f, 4f
      ))

      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 2)).fromArray(Array[Float](
        4f, 4f,
        4f, 4f
      ))
      // format: on

      val result = layer(input)
      result should approxEqual(expected, 1e-6f)

    it("should compute max pooling with asymmetrical window sizes (2x1)"):
      val layer = MaxPool2DLayer[H, W, Float32](
        window = (Axis[H] -> 2, Axis[W] -> 1),
        stride = 1,
        padding = Padding.VALID
      )

      // format: off
      val input = Tensor(Shape(Axis[H] -> 3, Axis[W] -> 3)).fromArray(Array[Float](
        1f, 2f, 3f,
        4f, 5f, 6f,
        7f, 8f, 9f
      ))

      val expected = Tensor(Shape(Axis[H] -> 2, Axis[W] -> 3)).fromArray(Array[Float](
        4f, 5f, 6f,
        7f, 8f, 9f
      ))
      // format: on

      val result = layer(input)
      result should approxEqual(expected, 1e-6f)
