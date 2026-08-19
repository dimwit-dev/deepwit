package deepwit.embedder

import deepwit.*
import dimwit.*
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class LearnedAbsolutePositionalInjectorSuite extends AnyFunSpec with Matchers:

  private val contextExtent = Axis[A] -> 3
  private val embeddingExtent = Axis[B] -> 2

  describe("LearnedAbsolutePositionalInjector"):

    it("adds the positional embeddings to the context"):
      // format: off
      val positional = Tensor(Shape(contextExtent, embeddingExtent)).fromArray(Array[Float](
        1f, 2f,
        3f, 4f,
        5f, 6f
      ))
      // format: on
      val injector = LearnedAbsolutePositionalInjector(LearnedAbsolutePositionalInjector.Params(positional))
      val context = Tensor(Shape(contextExtent, embeddingExtent)).fill(10f)
      injector(context) should approxEqual(context + positional, 1e-6f)

    it("is the identity for zero positional embeddings"):
      val zeros = Tensor(Shape(contextExtent, embeddingExtent)).fill(0f)
      val injector = LearnedAbsolutePositionalInjector(LearnedAbsolutePositionalInjector.Params(zeros))
      val context = Tensor(Shape(contextExtent, embeddingExtent)).fill(7f)
      injector(context) should approxEqual(context, 1e-6f)

    it("distinguishes positions"):
      val params = LearnedAbsolutePositionalInjector.Params.lecunNormal(contextExtent, embeddingExtent, Random.Key(42))
      val injector = LearnedAbsolutePositionalInjector(params)
      val context = Tensor(Shape(contextExtent, embeddingExtent)).fill(0f)
      val injected = injector(context)
      val first = injected.slice(Axis[A].at(0))
      val second = injected.slice(Axis[A].at(1))
      (first - second).abs.max.item should be > 0f

  describe("LearnedAbsolutePositionalInjector.Params"):

    it("lecunUniform has the requested shape and stays within its bounds"):
      // a = sqrt(3 / embeddingDim) = sqrt(3 / 4)
      val params = LearnedAbsolutePositionalInjector.Params.lecunUniform(Axis[A] -> 6, Axis[B] -> 4, Random.Key(42))
      params.positionalEmbeddings.shape(Axis[A]) shouldBe 6
      params.positionalEmbeddings.shape(Axis[B]) shouldBe 4
      params.positionalEmbeddings.abs.max.item should be <= math.sqrt(0.75).toFloat

    it("lecunNormal has the requested shape"):
      val params = LearnedAbsolutePositionalInjector.Params.lecunNormal(Axis[A] -> 6, Axis[B] -> 4, Random.Key(42))
      params.positionalEmbeddings.shape(Axis[A]) shouldBe 6
      params.positionalEmbeddings.shape(Axis[B]) shouldBe 4
