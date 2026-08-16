package deepwit.embedder

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class PositionalEncodingSuite extends AnyFunSpec with Matchers:

  private val shape = Shape(Axis[A] -> 4, Axis[B] -> 3, Axis[C] -> 8)

  describe("PositionalEncoding.sinusoidal2D"):

    it("has the shape it was asked for"):
      val encoding = PositionalEncoding.sinusoidal2D[A, B, C, Float32](shape)
      encoding.shape(Axis[A]) shouldBe 4
      encoding.shape(Axis[B]) shouldBe 3
      encoding.shape(Axis[C]) shouldBe 8

    it("stays within the range of sine and cosine"):
      val encoding = PositionalEncoding.sinusoidal2D[A, B, C, Float32](shape)
      encoding.abs.max.item should be <= 1f

    it("gives distinct encodings to distinct positions"):
      val encoding = PositionalEncoding.sinusoidal2D[A, B, C, Float32](shape)
      val at00 = encoding.slice(Axis[A].at(0)).slice(Axis[B].at(0))
      val at10 = encoding.slice(Axis[A].at(1)).slice(Axis[B].at(0))
      val at01 = encoding.slice(Axis[A].at(0)).slice(Axis[B].at(1))
      (at00 - at10).abs.max.item should be > 1e-3f
      (at00 - at01).abs.max.item should be > 1e-3f

    it("encodes the origin as sines of zero and cosines of one"):
      // Per axis the layout is [sin(scales), cos(scales)], and both axes are concatenated.
      val encoding = PositionalEncoding.sinusoidal2D[A, B, C, Float32](shape)
      val origin = encoding.slice(Axis[A].at(0)).slice(Axis[B].at(0))
      val expected = Tensor(Shape1(Axis[C] -> 8)).fromArray(Array(0f, 0f, 1f, 1f, 0f, 0f, 1f, 1f))
      origin should approxEqual(expected, 1e-6f)

    it("is deterministic"):
      val a = PositionalEncoding.sinusoidal2D[A, B, C, Float32](shape)
      val b = PositionalEncoding.sinusoidal2D[A, B, C, Float32](shape)
      a should approxEqual(b, 0f)

    it("requires the embedding dimension to be divisible by four"):
      an[IllegalArgumentException] should be thrownBy
        PositionalEncoding.sinusoidal2D[A, B, C, Float32](Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 6))
