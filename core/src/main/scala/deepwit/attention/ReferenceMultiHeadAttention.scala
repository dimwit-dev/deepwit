package deepwit.attention

import dimwit.*
import deepwit.base.AffineLayer
import dimwit.Label as Λ

/** TODO
  * This implementation is conceptually clearer than [[MultiHeadAttention]] but slower. How to merge?
  */

class ReferenceMultiHeadAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    params: ReferenceMultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, TargetEmbedding, V]) => Tensor2[Target, TargetEmbedding, V]):

  private val heads = params.heads.map(headParams => CustomAttention(headParams, createAttentionMask))
  private val projectHeadValues = AffineLayer(params.outputProjection)

  override def apply(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): Tensor2[Target, TargetEmbedding, V] =
    val headValues = stack(heads.map(head => head(source, target)), Axis[Head])
    headValues.vmap(Axis[Target])(headValues => projectHeadValues(headValues.flatten))

object ReferenceMultiHeadAttention:

  /** Holds the learnable parameters for a [[ReferenceMultiHeadAttention]].
    *
    * @param heads The parameters of one [[Attention]] per head.
    * @param outputProjection Projects the concatenated head values back into the target embedding space.
    */
  case class Params[SourceEmbedding, TargetEmbedding, V](
      heads: List[Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]],
      outputProjection: AffineLayer.Params[Head |*| HeadValue, TargetEmbedding, V]
  )

  object Params:

    def xavierUniformHeads[SourceEmbedding: Λ, TargetEmbedding: Λ, V: IsFloating](
        numHeads: Int,
        headQueryExtent: AxisExtent[HeadQuery],
        headKeyExtent: AxisExtent[HeadKey],
        headValueExtent: AxisExtent[HeadValue],
        sourceEmbeddingExtent: AxisExtent[SourceEmbedding],
        targetEmbeddingExtent: AxisExtent[TargetEmbedding],
        vtype: VType[V],
        key: Key
    ): List[Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, V]] =
      key
        .split(numHeads)
        .map(headKey => Attention.Params.init(headQueryExtent, headKeyExtent, headValueExtent, sourceEmbeddingExtent, targetEmbeddingExtent, vtype, headKey))
        .toList

    def xavierUniformOutputProjection[In: Λ, Out: Λ, V: IsFloating](numLayers: Int, embeddingExtent: AxisExtent[In], headExtent: AxisExtent[Out], vtype: VType[V], key: Key): AffineLayer.Params[In, Out, V] =
      val gain = Math.sqrt(1.0 / (2 * numLayers)).toFloat
      AffineLayer.Params.xavierUniform(embeddingExtent, headExtent, vtype, key, gain = gain)

    def xavierUniformDepthScaled[SourceEmbedding: Λ, TargetEmbedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, sourceEmbeddingExtent: AxisExtent[SourceEmbedding], targetEmbeddingExtent: AxisExtent[TargetEmbedding], vtype: VType[V], key: Key): Params[SourceEmbedding, TargetEmbedding, V] =
      require(targetEmbeddingExtent.size % numHeads == 0)
      require(sourceEmbeddingExtent.size % numHeads == 0)
      val (headsKey, projectionKey) = key.splitToTuple(2)
      val headExtent = Axis[Head] -> numHeads
      val headQueryExtent = Axis[HeadQuery] -> targetEmbeddingExtent.size / numHeads
      // The scaled dot-product contracts the query space against the key space, so the key extent follows the query extent.
      val headKeyExtent = Axis[HeadKey] -> targetEmbeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> sourceEmbeddingExtent.size / numHeads
      Params(
        heads = xavierUniformHeads(numHeads, headQueryExtent, headKeyExtent, headValueExtent, sourceEmbeddingExtent, targetEmbeddingExtent, vtype, headsKey),
        outputProjection = xavierUniformOutputProjection(numTransformerLayers, headExtent * headValueExtent, targetEmbeddingExtent, vtype, projectionKey)
      )
