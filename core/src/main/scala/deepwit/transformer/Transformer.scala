package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import dimwit.Label as Λ

/** A stack of [[TransformerLayer]]s followed by a final normalization.
  *
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param createAttentionMask A function generating a boolean mask to prevent attention to certain positions.
  */
class Transformer[Context: Λ, Embedding: Λ, V: IsFloating](
    params: Transformer.Params[Embedding, V],
    createAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val layers = params.transformerLayers.map(this.transformerLayer)
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(context_i)
    res.vmap(Axis[Context])(finalNorm)

  protected def transformerLayer(params: TransformerLayer.Params[Embedding, V]) = TransformerLayer(params, createAttentionMask)

object Transformer:

  /** A transformer that may only attend to preceding positions. */
  def causal[Context: Λ, Embedding: Λ, V: IsFloating](
      axis: Axis[Context],
      params: Params[Embedding, V]
  ): Transformer[Context, Embedding, V] = Transformer(params, causalMask[Context, Context])

  /** A transformer that may attend to all positions. */
  def bidirectional[Context: Λ, Embedding: Λ, V: IsFloating](
      axis: Axis[Context],
      params: Params[Embedding, V]
  ): Transformer[Context, Embedding, V] = Transformer(params, fullMask[Context, Context])

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
