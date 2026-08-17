package deepwit.attention

import dimwit.*
import dimwit.Label as Λ

/** Multi-head unrestricted attention as run by cuDNN's fused kernel.
  *
  * Fixed to [[BFloat16]] rather than generic in the element type, because that is the constraint the
  * kernel actually has — see [[FusedAttention]] for the rest of them. It computes what
  * [[MultiHeadFullAttention]] computes, and the two are required to agree.
  *
  * Note the head dimensions: cuDNN attends with the value space the key space has, so this only
  * covers parameters whose source and target embeddings are equally wide. Cross-attention between
  * differently sized embeddings stays on the head-by-head formulation.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
class MultiHeadFusedFullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]
) extends MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, BFloat16](sourceAxis, targetAxis, params):

  override protected def headValues(
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16]
  ): Tensor3[Head, Target, HeadValue, BFloat16] =
    FusedAttention.headValues(params, source, target, isCausal = false)
