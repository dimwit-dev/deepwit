package deepwit.regularization

import dimwit.*
import dimwit.stats.Bernoulli
import dimwit.Label as Λ

import deepwit.base.AffineLayer

/** Regularization by perturbing the parameters at random.
  *
  * Draw a random map on the parameters, measure the loss at the perturbed ones, and differentiate
  * through the map: what is minimized is then `E[loss(perturbed parameters)]`, an expectation over
  * an ensemble of models, estimated by one draw per step. Thinning — deleting feature directions
  * and rescaling what survives, known elsewhere as dropout — is one member of the family;
  * multiplicative Gaussian noise and additive weight noise are the same technique under a
  * different distribution. The randomness lives in the parameters handed to a model, never inside
  * it, so a model needs no random key and no train/eval flag, and every perturbation here is
  * mean-preserving so the untouched parameters can stand in at inference.
  *
  * Two things are easy to get wrong.
  *
  * '''Which weights.''' A feature is deleted by scaling the weights that ''read'' it, not the ones
  * that produce it: its consumer is linear in it, so the two are the same arithmetic for any
  * activation, whereas going through the producing weights would also need `φ(αz) = αφ(z)`, true
  * of ReLU but not of GELU, tanh or sigmoid. Hence an off-by-one — to delete what a hidden layer
  * produces, perturb the layer after it.
  *
  * '''Where it sits.''' The perturbation belongs inside the differentiated function, as
  * `valueAndGrad(params => cost(perturbed(params)))(params)`. Differentiating the plain cost ''at''
  * perturbed parameters is a different algorithm: it loses the factor the perturbation contributes
  * to the chain rule, so deleted weights collect gradients they should never receive.
  */
object Perturbation:

  /** Thins the feature space an affine layer reads: deletes each `In` direction with probability
    * `probability`, rescales the survivors by `1 / (1 - probability)` to keep the mean, and leaves
    * the bias alone. This is what is elsewhere called dropout on those features.
    *
    * To thin the feature space a hidden layer produces, apply this to the layer that consumes it.
    *
    * @param probability Must lie in `[0, 1)`. A scalar rather than a `Float` so it can be traced,
    *                    and scheduled the way a learning rate is.
    */
  def thin[In: Λ, Out: Λ, V: IsFloating](
      params: AffineLayer.Params[In, Out, V],
      probability: Tensor0[Float32],
      key: Random.Key
  ): AffineLayer.Params[In, Out, V] =
    val readScale = thinningDiagonal(params.weight.shape.extent(Axis[In]), probability, key)
    params.copy(weight = params.weight *! readScale.asFloat(VType[V]))

  /** One thinning of a feature space: zero where a direction was deleted, `1 / (1 - probability)`
    * where it survived. At a probability of zero this is the identity, by arithmetic rather than
    * by a branch around it.
    */
  private[regularization] def thinningDiagonal[L: Λ](extent: AxisExtent[L], probability: Tensor0[Float32], key: Random.Key): Tensor1[L, Float32] =
    val keepProbability = Tensor0(1f) - probability
    val ones = Tensor(Shape1(extent)).fill(1f)
    val kept = Bernoulli(Prob(ones.scale(keepProbability))).sample(key)
    where(kept, ones / ones.scale(keepProbability), Tensor(Shape1(extent)).fill(0f))
