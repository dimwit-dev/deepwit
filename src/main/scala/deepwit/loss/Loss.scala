package deepwit.loss

import dimwit.*

private def logsumexp[L: Label](logits: Tensor1[L, Float]): Tensor0[Float] =
  val maxLogit = logits.max(Axis[L])
  val logSumShifted = (logits -! maxLogit).exp.sum.log
  maxLogit + logSumShifted

def crossEntropy[L: Label](
    logits: Tensor1[L, Float],
    label: Tensor0[Int]
): Tensor0[Float] =
  val targetLogit = logits.slice(Axis[L].at(label))
  val logNormalizer = logsumexp(logits)
  logNormalizer - targetLogit
