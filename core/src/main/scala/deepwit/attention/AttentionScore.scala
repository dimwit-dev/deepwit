package deepwit.attention

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

/** Scores how strongly each target position attends to each source position, before masking. */
@FunctionalInterface
trait AttentionScore[Target, Source, Query, Key, V] extends ((Tensor2[Target, Query, V], Tensor2[Source, Key, V]) => Tensor2[Target, Source, V])

/** Default attention score functions from the literature. */
object AttentionScore:

  /** Computes the scaled dot-product attention scores between queries and keys as described in the original Transformer paper, [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
    *
    * $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
    */
  def scaledDotProduct[Target: Λ, Source: Λ, Query: Λ, Key: Λ, V: IsFloating](queries: Tensor2[Target, Query, V], keys: Tensor2[Source, Key, V]): Tensor2[Target, Source, V] =
    val dk = Math.sqrt(keys.shape(Axis[Key]))
    queries.dot(Axis[Query] -> Axis[Key])(keys) /! dk
