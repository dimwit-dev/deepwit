package deepwit.attention

import dimwit.*
import deepwit.base.AffineLayer
import deepwit.init
import dimwit.Label as Λ

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
abstract class MultiHeadAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, TargetEmbedding, V]) => Tensor2[Target, TargetEmbedding, V]):

  protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]

  private val projectHeadValues = AffineLayer(params.outputProjection)

  final def apply(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): Tensor2[Target, TargetEmbedding, V] =
    applyWithIntermediates(source, target).head

  final def applyWithIntermediates(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): (
      Tensor2[Target, TargetEmbedding, V],
      MultiHeadAttention.Intermediates[Source, Target, V]
  ) =
    val (heads, intermediates) = headValuesWithIntermediates(source, target)
    (projectHeads(heads), intermediates)

  protected def headValuesWithIntermediates(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): (
      Tensor3[Head, Target, HeadValue, V],
      MultiHeadAttention.Intermediates[Source, Target, V]
  ) =
    val (headValues, queries, keys, values) =
      zipvmap(Axis[Head])(params.queryWeights, params.keyWeights, params.valueWeights):
        case (q, k, v) =>
          val (headValue, perHead) = headAttention(Attention.Params(q, k, v)).applyWithIntermediates(source, target)
          (headValue, perHead.queries, perHead.keys, perHead.values)
    (headValues, (queries = queries, keys = keys, values = values))

  private def projectHeads(heads: Tensor3[Head, Target, HeadValue, V]): Tensor2[Target, TargetEmbedding, V] =
    heads.vmap(Axis[Target])(heads => projectHeadValues(heads.flatten))

object MultiHeadAttention:

  /** What every head attended from, stacked over the heads.
    *
    * The attention weights are deliberately not among them: they are the one intermediate that
    * costs a target by source matrix per head, and the one a fused kernel cannot report at all.
    */
  type Intermediates[Source, Target, V] = (
      queries: Tensor3[Head, Target, HeadQuery, V],
      keys: Tensor3[Head, Source, HeadKey, V],
      values: Tensor3[Head, Source, HeadValue, V]
  )

  case class Params[SourceEmbedding, TargetEmbedding, V](
      queryWeights: Tensor3[Head, TargetEmbedding, HeadQuery, V],
      keyWeights: Tensor3[Head, SourceEmbedding, HeadKey, V],
      valueWeights: Tensor3[Head, SourceEmbedding, HeadValue, V],
      outputProjection: AffineLayer.Params[Head |*| HeadValue, TargetEmbedding, V]
  )

  object Params:

    def xavierUniformHeads[Embedding: Λ, HeadOut: Λ, V: IsFloating](numHeads: Int, embeddingExtent: AxisExtent[Embedding], headExtent: AxisExtent[HeadOut], vtype: VType[V], key: Key): Tensor3[Head, Embedding, HeadOut, V] =
      stack(key.split(numHeads).map(key => init.xavierUniform(embeddingExtent, headExtent, vtype, key)), Axis[Head])

    def xavierUniformOutputProjection[In: Λ, Out: Λ, V: IsFloating](numLayers: Int, embeddingExtent: AxisExtent[In], headExtent: AxisExtent[Out], vtype: VType[V], key: Key): AffineLayer.Params[In, Out, V] =
      val gain = Math.sqrt(1.0 / (2 * numLayers)).toFloat
      AffineLayer.Params.xavierUniform(embeddingExtent, headExtent, vtype, key, gain = gain)

    def xavierUniformDepthScaled[SourceEmbedding: Λ, TargetEmbedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, sourceEmbeddingExtent: AxisExtent[SourceEmbedding], targetEmbeddingExtent: AxisExtent[TargetEmbedding], vtype: VType[V], key: Key): Params[SourceEmbedding, TargetEmbedding, V] =
      require(targetEmbeddingExtent.size % numHeads == 0)
      require(sourceEmbeddingExtent.size % numHeads == 0)
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val headExtent = Axis[Head] -> numHeads
      val headQueryExtent = Axis[HeadQuery] -> targetEmbeddingExtent.size / numHeads
      val headKeyExtent = Axis[HeadKey] -> targetEmbeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> sourceEmbeddingExtent.size / numHeads
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, targetEmbeddingExtent, headQueryExtent, vtype, queryKey),
        keyWeights = xavierUniformHeads(headExtent.size, sourceEmbeddingExtent, headKeyExtent, vtype, keyKey),
        valueWeights = xavierUniformHeads(headExtent.size, sourceEmbeddingExtent, headValueExtent, vtype, valueKey),
        outputProjection = xavierUniformOutputProjection(numTransformerLayers, headExtent * headValueExtent, targetEmbeddingExtent, vtype, projectionKey)
      )

/** Multi-head attention where every target position may attend to every source position.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
class MultiHeadFullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params):

  override protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V] =
    FullAttention(sourceAxis, targetAxis, headParams)

object MultiHeadFullAttention:

  /** Runs on [[MultiHeadFusedFullAttention]] wherever cuDNN accepts the parameters and the hardware,
    * and on the head-by-head formulation everywhere else.
    */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
  ): MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, V] =
    if FusedAttention.canRun(params) then
      // Guarded by the check above: it only passes when V is BFloat16.
      MultiHeadFusedFullAttention(sourceAxis, targetAxis, params.asInstanceOf[MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]])
        .asInstanceOf[MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]]
    else new MultiHeadFullAttention(sourceAxis, targetAxis, params)

/** Multi-head attention where a target position may only attend to source positions up to its own index.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
class MultiHeadCausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params):

  override protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V] =
    CausalAttention(sourceAxis, targetAxis, headParams)

object MultiHeadCausalAttention:

  /** Runs on [[MultiHeadFusedCausalAttention]] wherever cuDNN accepts the parameters and the
    * hardware, and on the head-by-head formulation everywhere else.
    */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
  ): MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, V] =
    if FusedAttention.canRun(params) then
      // Guarded by the check above: it only passes when V is BFloat16.
      MultiHeadFusedCausalAttention(sourceAxis, targetAxis, params.asInstanceOf[MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]])
        .asInstanceOf[MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]]
    else new MultiHeadCausalAttention(sourceAxis, targetAxis, params)

/** Multi-head attention restricted by a caller-supplied mask.
  *
  * @param mask A function generating a boolean mask to prevent attention to certain positions.
  */
class MultiHeadCustomAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    mask: Shape2[Target, Source] => Tensor2[Target, Source, Bool]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params):

  override protected def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V] =
    CustomAttention(headParams, mask)
