package deepwit.attention

import dimwit.*
import dimwit.hardware.DeviceBackend
import dimwit.jax.Jax
import dimwit.python.PyBridge
import dimwit.Label as Λ
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** Represents multi-head attention from a target sequence of embeddings onto a source sequence of embeddings.
  *
  * Every head runs an independent [[Attention]] on its own query, key and value space. The
  * concatenated head values are projected back into the target embedding space. Which source
  * positions a target position may attend to is left to the implementation:
  * [[MultiHeadFullAttention]], [[MultiHeadCausalAttention]] or [[MultiHeadCustomAttention]].
  *
  * Overriding [[headAttention]] is also how the heads get a different [[AttentionScore]].
  *
  * @tparam Source The axis label for the source sequence.
  * @tparam SourceEmbedding The axis label for the source embedding space.
  * @tparam Target The axis label for the target sequence.
  * @tparam TargetEmbedding The axis label for the target embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  */
trait MultiHeadUnfusedAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating]:
  this: MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V] =>
  protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]

  protected final def attend(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): (
      Tensor3[Target, Head, HeadValue, V],
      MultiHeadAttention.Intermediates[Source, Target, V]
  ) =
    val (headValues, queries, keys, values) =
      zipvmap(Axis[Head])(params.queryWeights, params.keyWeights, params.valueWeights):
        case (q, k, v) =>
          val (headValue, perHead) = headAttention(Attention.Params(q, k, v)).applyWithIntermediates(source, target)
          (headValue, perHead.queries, perHead.keys, perHead.values)
    (headValues.transpose(Axis[Target], Axis[Head], Axis[HeadValue]), (queries = queries.transpose(Axis[Target], Axis[Head], Axis[HeadQuery]), keys = keys.transpose(Axis[Source], Axis[Head], Axis[HeadKey]), values = values.transpose(Axis[Source], Axis[Head], Axis[HeadValue])))

/** Multi-head attention where every target position may attend to every source position.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
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

/** Multi-head attention where a target position may only attend to source positions up to its own index.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
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

/** Multi-head attention restricted by a caller-supplied mask.
  *
  * @param mask A function generating a boolean mask to prevent attention to certain positions.
  */
class MultiHeadCustomAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    mask: Shape2[Target, Source] => Tensor2[Target, Source, Bool],
    attentionScore: AttentionScore[Target, Source, HeadQuery, HeadKey, V]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params) with MultiHeadUnfusedAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]:

  override protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V] =
    CustomAttention(headParams, mask, attentionScore)
