package deepwit.attention

import dimwit.*
import dimwit.hardware.DeviceBackend
import dimwit.jax.Jax
import dimwit.python.PyBridge
import dimwit.Label as Λ
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** Marker trait for multi-head attention implementations that uses unfused, per-head attention. */
trait MultiHeadUnfusedAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating]:
  this: MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V] =>

  /** Implements the multi-head attention by attending head-by-head and concatenating the results. */
  protected final def attend(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): (
      Tensor3[Target, Head, HeadValue, V],
      MultiHeadAttention.Intermediates[Source, Target, V]
  ) =
    val (headValues, queries, keys, values) =
      zipvmap(Axis[Head])(params.queryWeights, params.keyWeights, params.valueWeights):
        case (q, k, v) =>
          val (headValue, perHead) = headAttention(Attention.Params(q, k, v)).applyWithIntermediates(source, target)
          (headValue, perHead.queries, perHead.keys, perHead.values)
    (headValues.swap(Axis[Target], Axis[Head]), (queries = queries.swap(Axis[Target], Axis[Head]), keys = keys.swap(Axis[Source], Axis[Head]), values = values.swap(Axis[Source], Axis[Head])))

  /** Per-head attention implementation. */
  protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]

// --- Unfused multi-head attention implementations ---

class MultiHeadUnfusedFullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]
) extends MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](sourceAxis, targetAxis, params) with MultiHeadUnfusedAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]:

  override protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V] =
    FullAttention(sourceAxis, targetAxis, headParams, attentionScore)

class MultiHeadUnfusedCausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]
) extends MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](sourceAxis, targetAxis, params) with MultiHeadUnfusedAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]:

  override protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V] =
    CausalAttention(sourceAxis, targetAxis, headParams, attentionScore)

class MultiHeadCustomAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    mask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
    attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params) with MultiHeadUnfusedAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]:

  override protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V] =
    CustomAttention(headParams, mask, attentionScore)
