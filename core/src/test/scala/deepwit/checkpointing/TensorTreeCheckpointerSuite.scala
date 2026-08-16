package deepwit.checkpointing

import java.nio.file.Files

import deepwit.*
import deepwit.base.AffineLayer
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorTreeCheckpointerSuite extends AnyFunSpec with Matchers:

  private def tempRoot(): String =
    val dir = Files.createTempDirectory("deepwit-checkpoint-test")
    dir.toFile.deleteOnExit()
    s"${dir.toString}/run"

  private def someParams = AffineLayer.Params.xavierUniform(Axis[A] -> 3, Axis[B] -> 2, VType[Float32], Random.Key(42))

  describe("TensorTreeCheckpointer"):

    it("round trips a parameter tree"):
      val checkpointer = TensorTreeCheckpointer(tempRoot())
      val params = someParams
      checkpointer.save(params, 0)
      val loaded = checkpointer.load[AffineLayer.Params[A, B, Float32]](0).get
      loaded.weight should approxEqual(params.weight, 0f)
      loaded.bias should approxEqual(params.bias, 0f)

    it("lists the saved iterations in ascending order"):
      val checkpointer = TensorTreeCheckpointer(tempRoot())
      val params = someParams
      List(20, 5, 100).foreach(step => checkpointer.save(params, step))
      checkpointer.iterations shouldBe Seq(5, 20, 100)

    it("reports no iterations before anything is saved"):
      TensorTreeCheckpointer(tempRoot()).iterations shouldBe empty

    it("rejects a negative iteration"):
      val checkpointer = TensorTreeCheckpointer(tempRoot())
      an[IllegalArgumentException] should be thrownBy checkpointer.save(someParams, -1)
      an[IllegalArgumentException] should be thrownBy checkpointer.load[AffineLayer.Params[A, B, Float32]](-1)

    it("refuses to write into an existing directory unless overwriting"):
      val root = tempRoot()
      TensorTreeCheckpointer(root).save(someParams, 0)
      an[AssertionError] should be thrownBy TensorTreeCheckpointer(root).save(someParams, 1)
      noException should be thrownBy TensorTreeCheckpointer(root, overwrite = true).save(someParams, 1)

    it("rejects an empty root path"):
      an[AssertionError] should be thrownBy TensorTreeCheckpointer("")
