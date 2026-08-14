package deepwit.transformer

import dimwit.*
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue}
import deepwit.normalization.LayerNorm
import dimwit.Label as Λ

class Transformer[Context: Λ, Embedding: Λ, V: IsFloating](
    params: Transformer.Params[Embedding, V],
    createAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val layers = params.transformerLayers.map(this.transformerLayer)
  private val postNorm = LayerNorm(params.postNorm)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(context_i)
    res.vmap(Axis[Context])(postNorm)

  protected def transformerLayer(params: TransformerLayer.Params[Embedding, V]) = TransformerLayer(params, createAttentionMask)

object Transformer:

  def causal[Context: Λ, Embedding: Λ, V: IsFloating](
      axis: Axis[Context],
      params: Params[Embedding, V]
  ): Transformer[Context, Embedding, V] = Transformer(params, causalMask[Context, Context])

  def bidirectional[Context: Λ, Embedding: Λ, V: IsFloating](
      axis: Axis[Context],
      params: Params[Embedding, V]
  ): Transformer[Context, Embedding, V] = Transformer(params, fullMask[Context, Context])

  case class Params[Embedding, V](
      transformerLayers: List[TransformerLayer.Params[Embedding, V]],
      postNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      new Params[Embedding, V](
        transformerLayers =
          key.split(numTransformerLayers).map: key =>
            TransformerLayer.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, embeddingExtent, embeddingMixedExtent, vtype, key)
          .toList,
        postNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
