package deepwit.embedder

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class PositionalEncodingSuite extends AnyFunSpec with Matchers:

  private val shape = Shape(Axis[A] -> 4, Axis[B] -> 3, Axis[C] -> 8)

  private def positions(values: Float*) =
    Tensor(Shape1(Axis[A] -> values.size)).fromArray(values.toArray)

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

  describe("PositionalEncoding.sinusoidal"):

    it("encodes a position the same way however densely it is sampled"):
      val coarse = PositionalEncoding.sinusoidal(positions(0f, 1f, 2f, 3f), Axis[C] -> 8)
      val fine = PositionalEncoding.sinusoidal(positions(0f, 0.5f, 1f, 1.5f, 2f, 2.5f, 3f, 3.5f), Axis[C] -> 8)
      coarse.slice(Axis[A].at(1)) should approxEqual(fine.slice(Axis[A].at(2)), 1e-6f)
      coarse.slice(Axis[A].at(3)) should approxEqual(fine.slice(Axis[A].at(6)), 1e-6f)

    it("accepts a position that lies between two grid indices"):
      val between = PositionalEncoding.sinusoidal(positions(0.5f), Axis[C] -> 8).slice(Axis[A].at(0))
      between.abs.max.item should be <= 1f

    it("spreads its slowest scale further as the frequency range grows"):
      val narrow = PositionalEncoding.sinusoidal(positions(0f, 8f), Axis[C] -> 8, frequencyRange = 10f)
      val wide = PositionalEncoding.sinusoidal(positions(0f, 8f), Axis[C] -> 8, frequencyRange = 10000f)
      val narrowTravel = (narrow.slice(Axis[A].at(0)) - narrow.slice(Axis[A].at(1))).abs.max.item
      val wideTravel = (wide.slice(Axis[A].at(0)) - wide.slice(Axis[A].at(1))).abs.max.item
      narrowTravel should be > wideTravel

    it("requires an even embedding dimension"):
      an[IllegalArgumentException] should be thrownBy
        PositionalEncoding.sinusoidal(positions(0f), Axis[C] -> 7)

    it("requires a frequency range above one"):
      an[IllegalArgumentException] should be thrownBy
        PositionalEncoding.sinusoidal(positions(0f), Axis[C] -> 8, frequencyRange = 1f)

  describe("PositionalEncoding.sinusoidal2D over given positions"):

    it("agrees with the grid form when handed the grid's own positions"):
      val vtype = VType[Float32]
      val fromGrid = PositionalEncoding.sinusoidal2D[A, B, C, Float32](shape)
      val fromPositions = PositionalEncoding.sinusoidal2D(
        PositionalEncoding.gridPositions(Axis[A] -> 4, vtype),
        PositionalEncoding.gridPositions(Axis[B] -> 3, vtype),
        Axis[C] -> 8
      )
      fromGrid should approxEqual(fromPositions, 0f)
