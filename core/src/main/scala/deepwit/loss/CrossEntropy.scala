package deepwit.loss

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

/** * Categorical (Multiclass) Cross Entropy Losses.
  * * Used for multi-class classification where the target is a discrete category
  * represented by an integer index.
  */
object CategoricalCrossEntropy:

  private def logsumexp[L: Λ, V: IsFloating](logits: Tensor1[L, V]): Tensor0[V] =
    val maxLogit = logits.max(Axis[L])
    val logSumShifted = (logits -! maxLogit).exp.sum.log
    maxLogit + logSumShifted

  /** * Computes Categorical Cross Entropy from probabilities.
    * * @param target The ground truth class index.
    * @param prediction A tensor of probabilities (e.g., from a Softmax).
    * @return Scalar loss value.
    */
  def apply[L: Λ, V: IsFloating](target: Tensor0[Int32], prediction: Tensor1[L, V]): Tensor0[V] =
    stable(ε = Tensor0(VType[V])(1e-7f))(target, prediction)

  /** * Computes Categorical Cross Entropy from raw unnormalized scores (logits).
    * * This version is more numerically stable than [[apply]] as it avoids
    * explicit Softmax and utilizes the Log-Sum-Exp identity.
    * * @param target The ground truth class index.
    * @param logits Raw model outputs in (-inf, +inf).
    */
  def fromLogits[L: Λ, V: IsFloating](target: Tensor0[Int32], logits: Tensor1[L, V]): Tensor0[V] =
    val targetLogit = logits.slice(Axis[L].at(target))
    val logNormalizer = logsumexp(logits)
    logNormalizer - targetLogit

  /** * Computes Categorical Cross Entropy with a custom stability epsilon.
    * * @param ε Small constant to prevent log(0).
    */
  def stable[L: Λ, V: IsFloating](ε: Tensor0[V])(target: Tensor0[Int32], prediction: Tensor1[L, V]): Tensor0[V] =
    val p = prediction.clip(ε, 1f - ε)
    -p.slice(Axis[L].at(target)).log

/** * Bernoulli Cross Entropy Losses for discrete binary labels.
  * * Specifically for classification tasks where the outcome is Boolean.
  * Internally treats True as 1.0 and False as 0.0.
  */
object BernoulliCrossEntropy:

  /** * Computes Bernoulli Cross Entropy from probabilities.
    * * @param target Discrete Boolean truth value.
    * @param prediction Probability of the positive class in [0, 1].
    */
  def apply[V: IsFloating](target: Tensor0[Bool], prediction: Tensor0[V]): Tensor0[V] =
    BinaryCrossEntropy.apply(target.asFloat(VType[V]), prediction)

  /** * Computes Bernoulli Cross Entropy from raw unnormalized scores (logits).
    * * @param target Discrete Boolean truth value.
    * @param logit Raw model output in (-inf, +inf).
    */
  def fromLogits[V: IsFloating](target: Tensor0[Bool], logit: Tensor0[V]): Tensor0[V] =
    BinaryCrossEntropy.fromLogits(target.asFloat(VType[V]), logit)

/** * General Binary Cross Entropy for soft/continuous targets.
  * * Useful for "Soft Bernoulli" distributions where the target is a probability
  * or intensity signal rather than a hard label (e.g., Image Reconstruction,
  * Label Smoothing).
  */
object BinaryCrossEntropy:

  /** * Computes Binary Cross Entropy from probabilities.
    * * @param target Target intensity/probability in [0, 1].
    * @param prediction Predicted probability in [0, 1].
    */
  def apply[V: IsFloating](target: Tensor0[V], prediction: Tensor0[V]): Tensor0[V] =
    stable(ε = Tensor0(VType[V])(1e-7f))(target, prediction)

  /** * Computes Binary Cross Entropy from raw unnormalized scores (logits).
    * * Uses a stable identity to prevent overflow/underflow:
    * `max(z, 0) - z*y + log(1 + exp(-|z|))`
    * * @param target Target intensity/probability in [0, 1].
    * @param logit Raw model output in (-inf, +inf).
    */
  def fromLogits[V: IsFloating](target: Tensor0[V], logit: Tensor0[V]): Tensor0[V] =
    maximum(logit, 0f) - (logit * target) + (1f + (-logit.abs).exp).log

  /** * Computes Binary Cross Entropy with a custom stability epsilon.
    * * @param ε Small constant to prevent log(0).
    */
  def stable[V: IsFloating](ε: Tensor0[V])(target: Tensor0[V], prediction: Tensor0[V]): Tensor0[V] =
    val p = prediction.clip(ε, 1f - ε)
    (target * -p.log) + ((1f - target) * -(1f - p).log)
