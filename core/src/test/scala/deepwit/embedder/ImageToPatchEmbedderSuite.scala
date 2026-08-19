package deepwit.embedder

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class ImageToPatchEmbedderSuite extends AnyFunSpec with Matchers:

  trait Width derives Label
  trait Height derives Label
  trait Channel derives Label
  trait PatchEmbedding derives Label

  private val patchExtent = 2
  private val embeddingExtent = Axis[PatchEmbedding] -> 4

  private def embedder =
    val params = ImageToPatchEmbedder.Params.xavierUniform(
      Axis[Width] -> patchExtent,
      Axis[Height] -> patchExtent,
      Axis[Channel] -> 1,
      embeddingExtent,
      Random.Key(42)
    )
    ImageToPatchEmbedder(params)

  private def image = Tensor(Shape(Axis[Width] -> 8, Axis[Height] -> 8, Axis[Channel] -> 1), VType[Float32]).fill(0.5f)

  describe("ImageToPatchEmbedder"):

    it("produces one embedded patch per patch of the image"):
      // An 8x8 image cut into 2x2 patches yields a 4x4 grid, flattened to 16 sequence elements.
      val patches = embedder(image)
      patches.shape(Axis[Width |*| Height]) shouldBe 16
      patches.shape(Axis[PatchEmbedding]) shouldBe 4

    it("is deterministic for the same parameters"):
      val fixed = embedder
      fixed(image) should approxEqual(fixed(image), 0f)

    it("distinguishes patches through the positional encoding"):
      // The image is constant, so any difference between patches comes from the 2D encoding.
      val patches = embedder(image)
      val first = patches.slice(Axis[Width |*| Height].at(0))
      val last = patches.slice(Axis[Width |*| Height].at(15))
      (first - last).abs.max.item should be > 1e-3f

  describe("ImageToPatchEmbedder.Params"):

    it("xavierUniform builds a kernel of the patch shape"):
      val params = ImageToPatchEmbedder.Params.xavierUniform(
        Axis[Width] -> 4,
        Axis[Height] -> 4,
        Axis[Channel] -> 3,
        embeddingExtent,
        Random.Key(42)
      )
      params.conv.kernel.shape(Axis[Width]) shouldBe 4
      params.conv.kernel.shape(Axis[Height]) shouldBe 4
      params.conv.kernel.shape(Axis[Channel]) shouldBe 3
      params.conv.kernel.shape(Axis[PatchEmbedding]) shouldBe 4
      params.conv.bias should approxEqual(Tensor(Shape1(embeddingExtent)).fill(0f), 1e-6f)
