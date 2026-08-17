package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import dimwit.Label as Λ

/** The GPT-2 transformer: a stack of [[TransformerLayer]]s followed by a final normalization.
  *
  * Causal throughout, as GPT-2 is. This is a composed architecture rather than a building block,
  * and is slated to move out of core.
  *
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param contextAxis The axis of the sequence attending onto itself.
  * @param params The learnable parameters.
  */
class Transformer[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: Transformer.Params[Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val layers = params.transformerLayers.map(this.transformerLayer)
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(context_i)
    res.vmap(Axis[Context])(finalNorm)

  protected def transformerLayer(params: TransformerLayer.Params[Embedding, V]) = TransformerLayer(contextAxis, params)

object Transformer:

  case class Params[Embedding, V](
      transformerLayers: List[TransformerLayer.Params[Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[Embedding, V] =
      new Params[Embedding, V](
        transformerLayers =
          key.split(numTransformerLayers).map: key =>
            TransformerLayer.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, embeddingMixedExtent, vtype, key)
          .toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
