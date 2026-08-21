package deepwit.attention

import dimwit.*
import deepwit.base.AffineLayer
import deepwit.init.Init
import dimwit.Label as Λ

/** Represents a multi-head attention mechanism from a target sequence of embeddings onto a source sequence of embeddings and projecting the result back to the target embedding space.
  *
  * $\text{MultiHeadAttention}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$ where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ (see [[Attention]] for the definition of $\text{Attention}$).
  */
abstract class MultiHeadAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    val params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, TargetEmbedding, V]) => Tensor2[Target, TargetEmbedding, V]):

  private val projectHeadValues = AffineLayer(params.outputProjection)

  final def apply(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): Tensor2[Target, TargetEmbedding, V] =
    applyWithIntermediates(source, target).head

  final def applyWithIntermediates(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): (
      Tensor2[Target, TargetEmbedding, V],
      MultiHeadAttention.Intermediates[Source, Target, V]
  ) =
    val (heads, intermediates) = attend(source, target)
    val res = heads.vmap(Axis[Target])(heads => projectHeadValues(heads.flatten))
    (res, intermediates)

  protected def attend(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): (
      Tensor3[Target, Head, HeadValue, V],
      MultiHeadAttention.Intermediates[Source, Target, V]
  )

object MultiHeadAttention:

  /** Holds the intermediate tensors computed during multi-head attention. */
  type Intermediates[Source, Target, V] = (
      queries: Tensor3[Target, Head, HeadQuery, V],
      keys: Tensor3[Source, Head, HeadKey, V],
      values: Tensor3[Source, Head, HeadValue, V]
  )

  /** Holds the parameters for the multi-head attention mechanism. */
  case class Params[SourceEmbedding, TargetEmbedding, V](
      queryWeights: Tensor3[Head, TargetEmbedding, HeadQuery, V],
      keyWeights: Tensor3[Head, SourceEmbedding, HeadKey, V],
      valueWeights: Tensor3[Head, SourceEmbedding, HeadValue, V],
      outputProjection: AffineLayer.Params[Head |*| HeadValue, TargetEmbedding, V]
  )

  object Params:

    def init[SourceEmbedding: Λ, TargetEmbedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, sourceEmbeddingExtent: AxisExtent[SourceEmbedding], targetEmbeddingExtent: AxisExtent[TargetEmbedding], key: Key, vtype: VType[V] = VType[Float32]): Params[SourceEmbedding, TargetEmbedding, V] =
      xavierUniformDepthScaled(numTransformerLayers, numHeads, sourceEmbeddingExtent, targetEmbeddingExtent, key, vtype)

    def xavierUniformDepthScaled[SourceEmbedding: Λ, TargetEmbedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, sourceEmbeddingExtent: AxisExtent[SourceEmbedding], targetEmbeddingExtent: AxisExtent[TargetEmbedding], key: Key, vtype: VType[V] = VType[Float32]): Params[SourceEmbedding, TargetEmbedding, V] =
      require(targetEmbeddingExtent.size % numHeads == 0)
      require(sourceEmbeddingExtent.size % numHeads == 0)

      def xavierUniformHeads[Embedding: Λ, HeadOut: Λ, V: IsFloating](numHeads: Int, embeddingExtent: AxisExtent[Embedding], headExtent: AxisExtent[HeadOut], key: Key, vtype: VType[V] = VType[Float32]): Tensor3[Head, Embedding, HeadOut, V] =
        stack(key.split(numHeads).map(key => Init.xavierUniform(embeddingExtent, headExtent, key, vtype)), Axis[Head])

      def xavierUniformOutputProjection[In: Λ, Out: Λ, V: IsFloating](numLayers: Int, embeddingExtent: AxisExtent[In], headExtent: AxisExtent[Out], key: Key, vtype: VType[V] = VType[Float32]): AffineLayer.Params[In, Out, V] =
        val gain = Math.sqrt(1.0 / (2 * numLayers)).toFloat
        AffineLayer.Params.xavierUniform(embeddingExtent, headExtent, key, vtype, gain = gain)

      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val headExtent = Axis[Head] -> numHeads
      val headQueryExtent = Axis[HeadQuery] -> targetEmbeddingExtent.size / numHeads
      val headKeyExtent = Axis[HeadKey] -> targetEmbeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> sourceEmbeddingExtent.size / numHeads
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, targetEmbeddingExtent, headQueryExtent, queryKey, vtype),
        keyWeights = xavierUniformHeads(headExtent.size, sourceEmbeddingExtent, headKeyExtent, keyKey, vtype),
        valueWeights = xavierUniformHeads(headExtent.size, sourceEmbeddingExtent, headValueExtent, valueKey, vtype),
        outputProjection = xavierUniformOutputProjection(numTransformerLayers, headExtent * headValueExtent, targetEmbeddingExtent, projectionKey, vtype)
      )

/** Multi-head attention where every target position may attend to every source position. */
abstract class MultiHeadFullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params)

object MultiHeadFullAttention:

  /** Smart constructor that routes to fused or unfused attention depending on the parameters and hardware for best out-the-box efficiency. */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
  ): MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, V] =
    if MultiHeadFusedFullAttention.isAvailable(VType[V].dtype) then
      val paramsBFloat16 = params.asInstanceOf[MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]] // Guarded by the check above
      val res = MultiHeadFusedFullAttention(sourceAxis, targetAxis, paramsBFloat16)
      res.asInstanceOf[MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]]
    else MultiHeadUnfusedFullAttention(sourceAxis, targetAxis, params, AttentionScore.scaledDotProduct)

/** Multi-head attention where a target position may only attend to source positions up to its own index. */
abstract class MultiHeadCausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
) extends MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, V](params)

object MultiHeadCausalAttention:

  /** Smart constructor that routes to fused or unfused attention depending on the parameters and hardware for best out-the-box efficiency. */
  def apply[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
      sourceAxis: Axis[Source],
      targetAxis: Axis[Target],
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
  ): MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, V] =
    if MultiHeadFusedCausalAttention.isAvailable(VType[V].dtype) then
      val paramsBFloat16 = params.asInstanceOf[MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]] // Guarded by the check above
      val res = MultiHeadFusedCausalAttention(sourceAxis, targetAxis, paramsBFloat16)
      res.asInstanceOf[MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, V]]
    else MultiHeadUnfusedCausalAttention(sourceAxis, targetAxis, params, AttentionScore.scaledDotProduct)
