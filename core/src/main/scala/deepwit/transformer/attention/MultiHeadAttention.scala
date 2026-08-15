package deepwit.transformer.attention

import dimwit.*
import deepwit.base.AffineLayer
import deepwit.init
import dimwit.Label as Λ

/** Represents multi-head attention from a target sequence of embeddings onto a source sequence of embeddings.
  *
  * Every head runs an independent [[Attention]] on its own query, key and value space. The
  * concatenated head values are projected back into the target embedding space.
  *
  * @tparam Source The axis label for the source sequence.
  * @tparam SourceEmbedding The axis label for the source embedding space.
  * @tparam Target The axis label for the target sequence.
  * @tparam TargetEmbedding The axis label for the target embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param createAttentionMask A function generating a boolean mask to prevent attention to certain positions.
  */
class MultiHeadAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ, V: IsFloating](
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V],
    createAttentionMask: Shape2[Target, Source] => Tensor2[Target, Source, Bool]
) extends ((Tensor2[Source, SourceEmbedding, V], Tensor2[Target, TargetEmbedding, V]) => Tensor2[Target, TargetEmbedding, V]):

  private val projectHeadValues = AffineLayer(params.outputProjection)

  override def apply(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]): Tensor2[Target, TargetEmbedding, V] =
    val heads = headAttention(source, target)
    heads.vmap(Axis[Target])(heads => projectHeadValues(heads.flatten))

  protected def headAttention(source: Tensor2[Source, SourceEmbedding, V], target: Tensor2[Target, TargetEmbedding, V]) =
    zipvmap(Axis[Head])(params.queryWeights, params.keyWeights, params.valueWeights):
      case (q, k, v) =>
        val headAttention = Attention(Attention.Params(q, k, v), createAttentionMask)
        headAttention(source, target)

object MultiHeadAttention:

  case class Params[SourceEmbedding, TargetEmbedding, V](
      queryWeights: Tensor3[Head, TargetEmbedding, HeadQuery, V], // TODO update to List[Attention.Params] => Check performance
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
      // The scaled dot-product contracts the query space against the key space, so the key extent follows the query extent.
      val headKeyExtent = Axis[HeadKey] -> targetEmbeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> sourceEmbeddingExtent.size / numHeads
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, targetEmbeddingExtent, headQueryExtent, vtype, queryKey),
        keyWeights = xavierUniformHeads(headExtent.size, sourceEmbeddingExtent, headKeyExtent, vtype, keyKey),
        valueWeights = xavierUniformHeads(headExtent.size, sourceEmbeddingExtent, headValueExtent, vtype, valueKey),
        outputProjection = xavierUniformOutputProjection(numTransformerLayers, headExtent * headValueExtent, targetEmbeddingExtent, vtype, projectionKey)
      )
