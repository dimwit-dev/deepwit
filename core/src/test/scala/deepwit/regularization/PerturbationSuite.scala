package deepwit.regularization

import deepwit.*
import dimwit.*

import deepwit.base.AffineLayer
import deepwit.activation.gelu
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class PerturbationSuite extends AnyFunSpec with Matchers:

  trait In derives Label
  trait Hidden derives Label
  trait Out derives Label

  private val inExtent = Axis[In] -> 4
  private val hiddenExtent = Axis[Hidden] -> 6
  private val outExtent = Axis[Out] -> 2

  private val layer = AffineLayer.Params.init(hiddenExtent, outExtent, Random.Key(1))
  private val probability = Tensor0(0.5f)

  /** Which directions a diagonal lets through, as a mask of ones and zeros. */
  private def survivors(diagonal: Tensor1[Hidden, Float32]): Seq[Float] =
    (0 until hiddenExtent.size).map(i => if diagonal.slice(Axis[Hidden].at(i)).item == 0f then 0f else 1f)

  private def readScaleOf(thinned: AffineLayer.Params[Hidden, Out, Float32], i: Int): Seq[Float] =
    (0 until outExtent.size).map: o =>
      val before = layer.weight.slice(Axis[Hidden].at(i)).slice(Axis[Out].at(o)).item
      val after = thinned.weight.slice(Axis[Hidden].at(i)).slice(Axis[Out].at(o)).item
      if before == 0f then 0f else after / before

  describe("Perturbation.thinningDiagonal"):

    it("deletes directions and rescales the survivors by one over the keep probability"):
      val diagonal = Perturbation.thinningDiagonal(hiddenExtent, probability, Random.Key(42))
      val values = (0 until hiddenExtent.size).map(i => diagonal.slice(Axis[Hidden].at(i)).item)
      values should contain(0f) // this draw deleted something, so there is a deletion to look at
      values.filter(_ != 0f).distinct shouldBe Seq(2f) // survivors scaled by 1 / (1 - 0.5)

    it("is exactly the identity at a probability of zero, by arithmetic rather than a branch"):
      val diagonal = Perturbation.thinningDiagonal(hiddenExtent, Tensor0(0f), Random.Key(7))
      (0 until hiddenExtent.size).foreach: i =>
        diagonal.slice(Axis[Hidden].at(i)).item shouldBe 1f

  describe("Perturbation.thin"):

    it("scales each of a layer's input directions by the diagonal, and leaves the bias alone"):
      val key = Random.Key(42)
      val thinned = Perturbation.thin(layer, probability, key)
      val diagonal = Perturbation.thinningDiagonal(hiddenExtent, probability, key)
      (0 until hiddenExtent.size).foreach: i =>
        val expected = diagonal.slice(Axis[Hidden].at(i)).item
        readScaleOf(thinned, i).foreach(_ shouldBe expected +- 1e-5f)
      thinned.bias should approxEqual(layer.bias, 1e-6f)

    it("matches masking the activations directly, under an activation that is not homogeneous"):
      // The claim the whole formulation rests on. Deleting a feature by zeroing the weights that
      // PRODUCE it needs φ(αz) = αφ(z), which gelu does not satisfy. Mutating the weights that
      // READ it needs nothing of φ at all, because whatever consumes a feature does so linearly.
      val key = Random.Key(42)
      val producing = AffineLayer.Params.init(inExtent, hiddenExtent, Random.Key(2))
      val x = Tensor(Shape1(inExtent)).fromArray(Array(0.7f, -1.3f, 0.2f, 0.9f))
      val h = gelu(AffineLayer(producing)(x))

      val diagonal = Perturbation.thinningDiagonal(hiddenExtent, probability, key)
      val maskedActivations = AffineLayer(layer)(h * diagonal)
      val thinnedWeights = AffineLayer(Perturbation.thin(layer, probability, key))(h)

      survivors(diagonal) should contain(0f) // the draw really deleted something
      thinnedWeights should approxEqual(maskedActivations, 1e-5f)

    it("sends no gradient into a deleted direction when differentiated through"):
      // The thinning must sit INSIDE the differentiated function. Differentiating the thinned tree
      // and applying that gradient to the stored one would lose the factor the thinning
      // contributes, and deleted weights would collect updates they should never receive.
      val key = Random.Key(42)
      val h = Tensor(Shape1(hiddenExtent)).fill(1f)
      def cost(params: AffineLayer.Params[Hidden, Out, Float32]): Tensor0[Float32] =
        AffineLayer(Perturbation.thin(params, probability, key))(h).sum
      val gradient = Autodiff.grad(cost)(layer).value
      val diagonal = Perturbation.thinningDiagonal(hiddenExtent, probability, key)
      survivors(diagonal).zipWithIndex.foreach: (survived, i) =>
        val g = gradient.weight.slice(Axis[Hidden].at(i)).slice(Axis[Out].at(0)).item
        if survived == 0f then g shouldBe 0f else g shouldBe 2f +- 1e-5f
