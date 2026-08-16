package deepwit.init

import deepwit.*
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class InitializationSuite extends AnyFunSpec with Matchers:

  describe("xavierUniform"):

    it("has the requested shape"):
      val w = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      w.shape(Axis[A]) shouldBe 4
      w.shape(Axis[B]) shouldBe 6

    it("stays within the Glorot bounds"):
      // a = sqrt(3 * 2 / (4 + 6)) = sqrt(0.6)
      val a = math.sqrt(0.6).toFloat
      val w = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      w.abs.max.item should be <= a

    it("scales linearly with the gain"):
      val plain = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      val scaled = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42), gain = 2f)
      scaled should approxEqual(plain *! Tensor0(2f), 1e-6f)

  describe("xavierNormal"):

    it("has the requested shape"):
      val w = xavierNormal(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      w.shape(Axis[A]) shouldBe 4
      w.shape(Axis[B]) shouldBe 6

    it("has zero mean and the expected standard deviation"):
      // variance = 2 / (100 + 100) = 0.01 => std = 0.1
      val w = xavierNormal(Axis[A] -> 100, Axis[B] -> 100, VType[Float32], Random.Key(42))
      w.mean.item shouldBe (0f +- 0.005f)
      val std = (w -! w.mean).pow(2).mean.sqrt
      std.item shouldBe (0.1f +- 0.005f)

    it("scales linearly with the gain"):
      val plain = xavierNormal(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      val scaled = xavierNormal(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42), gain = 2f)
      scaled should approxEqual(plain *! Tensor0(2f), 1e-6f)

  describe("the vector variants"):

    it("treat the fan-out as one"):
      // a = sqrt(3 * 2 / (5 + 1)) = 1
      val w = xavierUniformVector(Axis[A] -> 5, VType[Float32], Random.Key(42))
      w.shape(Axis[A]) shouldBe 5
      w.abs.max.item should be <= 1f

    it("xavierNormalVector has the expected standard deviation"):
      // variance = 2 / (1000 + 1) => std = sqrt(0.001998)
      val expectedStd = math.sqrt(2.0 / 1001).toFloat
      val w = xavierNormalVector(Axis[A] -> 1000, VType[Float32], Random.Key(42))
      val std = (w -! w.mean).pow(2).mean.sqrt
      std.item shouldBe (expectedStd +- 0.005f)

  describe("determinism"):

    it("gives the same weights for the same key"):
      val a = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      val b = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      a should approxEqual(b, 0f)

    it("gives different weights for different keys"):
      val a = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(42))
      val b = xavierUniform(Axis[A] -> 4, Axis[B] -> 6, VType[Float32], Random.Key(43))
      (a - b).abs.max.item should be > 0f
