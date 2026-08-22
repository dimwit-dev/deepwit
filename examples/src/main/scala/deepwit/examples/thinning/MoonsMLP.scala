package deepwit.examples.thinning

import dimwit.*

import deepwit.examples.dataset.TwoMoons
import deepwit.examples.dataset.TwoMoons.{Feature, Output}

import deepwit.base.AffineLayer
import deepwit.activation.relu
import deepwit.regularization.Perturbation

/** A two-hidden-layer classifier for the two moons.
  *
  * The model is not regularized: no random key, no train/eval flag, nothing in its type that knows
  * regularization exists. That lives entirely in [[MoonsMLP.Params.thin]], which hands back a
  * different parameter set. See the README in this directory.
  */
class MoonsMLP(params: MoonsMLP.Params) extends (Tensor1[Feature, Float32] => Tensor0[Int32]):

  private val layer1 = AffineLayer(params.layer1)
  private val layer2 = AffineLayer(params.layer2)
  private val output = AffineLayer(params.output)

  override def apply(point: Tensor1[Feature, Float32]): Tensor0[Int32] =
    logits(point).argmax(Axis[Output])

  def logits(point: Tensor1[Feature, Float32]): Tensor1[Output, Float32] =
    output(relu(layer2(relu(layer1(point)))))

object MoonsMLP:

  def apply(params: Params): MoonsMLP = new MoonsMLP(params)

  case class Params(
      layer1: AffineLayer.Params[Feature, Hidden1, Float32],
      layer2: AffineLayer.Params[Hidden1, Hidden2, Float32],
      output: AffineLayer.Params[Hidden2, Output, Float32]
  ):

    /** One member of the ensemble: the same model with a random fifth of each hidden layer's
      * features deleted and the survivors rescaled.
      *
      * Note the off-by-one. Dropping what `layer1` produces means thinning what reads it, which is
      * `layer2`. Apply this inside the function being differentiated; see [[Perturbation]].
      */
    def thin(probability: Tensor0[Float32], key: Key): Params =
      val (hidden1Key, hidden2Key) = key.split2()
      copy(
        layer2 = Perturbation.thin(layer2, probability, hidden1Key), // deletes hidden1 features
        output = Perturbation.thin(output, probability, hidden2Key)  // deletes hidden2 features
      )

  object Params:

    def init(hiddenSize: Int, key: Key): Params =
      val (layer1Key, layer2Key, outputKey) = key.splitToTuple(3)
      val hidden1Extent = Axis[Hidden1] -> hiddenSize
      val hidden2Extent = Axis[Hidden2] -> hiddenSize
      Params(
        layer1 = AffineLayer.Params.init(TwoMoons.featureExtent, hidden1Extent, layer1Key),
        layer2 = AffineLayer.Params.init(hidden1Extent, hidden2Extent, layer2Key),
        output = AffineLayer.Params.init(hidden2Extent, TwoMoons.outputExtent, outputKey)
      )
