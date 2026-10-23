package deepwit.attention

import dimwit.*
import dimwit.Label as Λ

/** Marker trait for multi-head attention implementations that uses unfused, per-head attention. */
trait MultiHeadUnfusedAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating]:
  this: MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V] =>

  /** Implements the multi-head attention by attending head-by-head, over every position of the target and the source. */
  protected final def attend(queries: Tensor3[Target, Head, HeadQuery, V], keys: Tensor3[Source, Head, HeadKey, V], values: Tensor3[Source, Head, HeadValue, V]): Tensor3[Target, Head, HeadValue, V] =
    val targetPositions = Tensor1(Axis[Target]).fromRange(0 until queries.shape(Axis[Target]))
    val sourcePositions = Tensor1(Axis[Source]).fromRange(0 until keys.shape(Axis[Source]))
    attendAt(queries, keys, values, targetPositions, sourcePositions)

  /** Attends head by head from the `queries` at `targetPositions` to the `keys` and `values` at `sourcePositions`.
    *
    * Decoding with a [[KVCache]] attends from a single target position to every position of the cache.
    */
  final def attendAt(
      queries: Tensor3[Target, Head, HeadQuery, V],
      keys: Tensor3[Source, Head, HeadKey, V],
      values: Tensor3[Source, Head, HeadValue, V],
      targetPositions: Tensor1[Target, Int32],
      sourcePositions: Tensor1[Source, Int32]
  ): Tensor3[Target, Head, HeadValue, V] =
    val attended = zipvmap(Axis[Head])(queries, keys, values):
      case (headQueries, headKeys, headValues) =>
        Attention.attend(attentionScore, attentionMask)(headQueries, headKeys, headValues, targetPositions, sourcePositions)
    attended.swap(Axis[Head], Axis[Target])

  /** How every head scores its queries against its keys. */
  protected def attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]

  /** Which source positions every head lets each target position attend to. */
  protected def attentionMask: AttentionMask[Target, Source]

// --- Unfused multi-head attention implementations ---

class MultiHeadUnfusedFullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    protected val attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]
) extends MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](sourceAxis, targetAxis, params) with MultiHeadUnfusedAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]:

  override protected def attentionMask: AttentionMask[Target, Source] = fullMask

class MultiHeadUnfusedCausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    protected val attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]
) extends MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](sourceAxis, targetAxis, params) with MultiHeadUnfusedAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]:

  override protected def attentionMask: AttentionMask[Target, Source] = causalMask

class MultiHeadCustomAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    mask: AttentionMask[Target, Source],
    protected val attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params) with MultiHeadUnfusedAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]:

  override protected def attentionMask: AttentionMask[Target, Source] = mask
