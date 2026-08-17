package deepwit.transformer

import dimwit.*
import dimwit.Label as Λ
import deepwit.normalization.LayerNorm

/** A stack of [[CrossTransformerLayer]]s followed by a final normalization.
  *
  * Unrestricted in both directions, as its layers are. This is a composed architecture rather than
  * a building block, and is slated to move out of core.
  *
  * @tparam CrossContext The axis label for the cross sequence.
  * @tparam CrossEmbedding The axis label for the cross embedding space.
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param crossContextAxis The axis of the sequence being attended onto.
  * @param contextAxis The axis of the sequence attending onto itself and onto the cross context.
  * @param params The learnable parameters.
  */
class CrossTransformer[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    crossContextAxis: Axis[CrossContext],
    contextAxis: Axis[Context],
    params: CrossTransformer.Params[CrossEmbedding, Embedding, V]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  private val layers = params.transformerLayers.map(p => CrossTransformerLayer(crossContextAxis, contextAxis, p))
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
