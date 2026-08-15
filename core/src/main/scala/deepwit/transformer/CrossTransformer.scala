package deepwit.transformer

import dimwit.*
import dimwit.Label as Λ
import deepwit.normalization.LayerNorm

/** A stack of [[CrossTransformerLayer]]s followed by a final normalization.
  *
  * @tparam CrossContext The axis label for the cross sequence.
  * @tparam CrossEmbedding The axis label for the cross embedding space.
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param createCrossAttentionMask A function generating a boolean mask for the cross-attention.
  * @param createSelfAttentionMask A function generating a boolean mask for the self-attention.
  */
class CrossTransformer[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    params: CrossTransformer.Params[CrossEmbedding, Embedding, V],
    createCrossAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Bool],
    createSelfAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  private val layers = params.transformerLayers.map(p => CrossTransformerLayer(p, createCrossAttentionMask, createSelfAttentionMask))
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)
    res.vmap(Axis[Context])(finalNorm)

  def applyWithHiddenStates(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): (List[Tensor2[Context, Embedding, V]], Tensor2[Context, Embedding, V]) =
    val allStates = layers.scanLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)
    val hiddenStates = allStates.tail // drop initial context
    val res = hiddenStates.last
    (hiddenStates, res.vmap(Axis[Context])(finalNorm))

object CrossTransformer:

  /** A cross transformer in decoder configuration: causal over its own context, unrestricted onto the cross context. */
  def decoder[CrossContext: Λ, Context: Λ, CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](
      crossAxis: Axis[CrossContext],
      axis: Axis[Context],
      params: CrossTransformer.Params[CrossEmbedding, Embedding, V]
  ): CrossTransformer[CrossContext, CrossEmbedding, Context, Embedding, V] = CrossTransformer(
    params,
    createCrossAttentionMask = fullMask[Context, CrossContext],
    createSelfAttentionMask = causalMask[Context, Context]
  )

  case class Params[CrossEmbedding, Embedding, V](
      transformerLayers: List[CrossTransformerLayer.Params[CrossEmbedding, Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[CrossEmbedding, Embedding, V] =
      new Params(
        transformerLayers =
          key.split(numTransformerLayers).map: key =>
            CrossTransformerLayer.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, crossEmbeddingExtent, embeddingExtent, embeddingMixedExtent, vtype, key)
          .toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
