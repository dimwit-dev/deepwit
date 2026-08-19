package deepwit.regularization

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Bernoulli
import dimwit.Label as Λ

/** Dropout as a projection of a feature space onto itself.
  *
  * The projection is the identity, so the layer is a no-op wherever it sits. Dropout happens to the
  * parameters rather than to the activations: [[Dropout.Params.thinned]] deletes a random set of
  * feature directions and rescales the survivors. The layer therefore needs no random key and no
  * train/eval flag — only the training loop does, to thin the parameters it is about to
  * differentiate.
  *
  * Deleting a direction can only scale a feature by zero or by one over the keep probability, so
  * the projection stays diagonal and is held as that diagonal.
  *
  * @tparam Feature The axis label for the feature space being projected onto itself.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The projection, identity until thinned.
  */
class Dropout[Feature: Λ, V: IsFloating](params: Dropout.Params[Feature, V]) extends (Tensor1[Feature, V] => Tensor1[Feature, V]):

  override def apply(features: Tensor1[Feature, V]): Tensor1[Feature, V] =
    features * params.scale

object Dropout:

  def apply[Feature: Λ, V: IsFloating](params: Params[Feature, V]): Dropout[Feature, V] = new Dropout(params)

  /** Holds the projection of a [[Dropout]], as the diagonal it is: one scale per feature. */
  case class Params[Feature, V](scale: Tensor1[Feature, V]):

    /** Deletes each feature direction with probability `probability` and rescales those that remain.
      *
      * Scaling the survivors by `1 / (1 - probability)` keeps the expected output equal to the
      * input, so the layer can be left as the identity at evaluation time. Thin the stored identity
      * afresh each step and keep the result out of the optimizer.
      */
    def thinned(probability: Float, key: Random.Key)(using Λ[Feature], IsFloating[V]): Params[Feature, V] =
      require(probability >= 0f && probability < 1f, s"A dropout probability must lie in [0, 1), but was $probability.")
      val keepProbability = 1f - probability
      val kept = Bernoulli(Prob(Tensor(scale.shape).fill(keepProbability))).sample(key)
      val survivorScale = Tensor.like(scale).fill(1f / keepProbability)
      Params(scale * where(kept, survivorScale, Tensor.like(scale).fill(0f)))

  object Params:

    def init[Feature: Λ, V: IsFloating](featureExtent: AxisExtent[Feature], vtype: VType[V] = VType[Float32]): Params[Feature, V] =
      identity(featureExtent, vtype)

    def identity[Feature: Λ, V: IsFloating](featureExtent: AxisExtent[Feature], vtype: VType[V] = VType[Float32]): Params[Feature, V] =
      Params(Tensor(Shape1(featureExtent), vtype).fill(1f))
