package deepwit.checkpointing

import java.nio.file.{Files, Path}

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

  private def someParams = AffineLayer.Params.xavierUniform(Axis[A] -> 3, Axis[B] -> 2, Random.Key(42))

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

    it("loads the furthest iteration reached"):
      val checkpointer = TensorTreeCheckpointer(tempRoot())
      List(20, 5, 100).foreach(step => checkpointer.save(someParams, step))
      checkpointer.iterations.lastOption shouldBe Some(100)
      checkpointer.loadLatest[AffineLayer.Params[A, B, Float32]] should not be empty

    it("has no latest checkpoint before anything is saved"):
      TensorTreeCheckpointer(tempRoot()).loadLatest[AffineLayer.Params[A, B, Float32]] shouldBe None

  describe("TensorTreeCheckpointer.newIn"):

    it("puts the run in its own folder under the given root"):
      val root = tempRoot()
      val checkpointer = TensorTreeCheckpointer.newIn(root)
      checkpointer.rootPath should startWith(s"$root/")
      checkpointer.save(someParams, 0)
      Path.of(checkpointer.rootPath).toFile.isDirectory shouldBe true

  describe("TensorTreeCheckpointer.latestIn"):

    it("reads the run whose name sorts last"):
      val root = tempRoot()
      List("20240101_000000", "20260823_120000", "20250601_083000").foreach: name =>
        TensorTreeCheckpointer(s"$root/$name").save(someParams, 7)
      TensorTreeCheckpointer.latestIn(root).get.rootPath shouldBe s"$root/20260823_120000"

    it("ignores modification times, which a copy or a re-save would disturb"):
      val root = tempRoot()
      List("20240101_000000", "20260823_120000").foreach: name =>
        TensorTreeCheckpointer(s"$root/$name").save(someParams, 7)
      // Touch the older run, as restoring or re-saving it would.
      Path.of(s"$root/20240101_000000").toFile.setLastModified(System.currentTimeMillis() + 100000)
      TensorTreeCheckpointer.latestIn(root).get.rootPath shouldBe s"$root/20260823_120000"

    it("is empty when no run has been written yet"):
      TensorTreeCheckpointer.latestIn(tempRoot()) shouldBe None

    it("passes over folders that are not runs"):
      val root = tempRoot()
      TensorTreeCheckpointer(s"$root/20260823_120000").save(someParams, 7)
      List("dataset", "zzz-notes", "20260824_120000_backup").foreach: name =>
        Files.createDirectories(Path.of(s"$root/$name"))
      TensorTreeCheckpointer.latestIn(root).get.rootPath shouldBe s"$root/20260823_120000"

    it("is empty when a root holds folders but no run"):
      val root = tempRoot()
      Files.createDirectories(Path.of(s"$root/dataset"))
      TensorTreeCheckpointer.latestIn(root) shouldBe None
