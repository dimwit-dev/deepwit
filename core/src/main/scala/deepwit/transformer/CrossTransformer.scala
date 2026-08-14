package deepwit.transformer

import dimwit.*
import dimwit.Label as Λ
import deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue}
import deepwit.normalization.LayerNorm

case class CrossTransformer[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    params: CrossTransformer.Params[CrossEmbedding, Embedding, V],
    createCrossAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Bool],
    createSelfAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  private val layers = params.transformerLayers.map(p => CrossTransformerLayer(p, createCrossAttentionMask, createSelfAttentionMask))
  private val postNorm = LayerNorm(params.postNorm)

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)
    res.vmap(Axis[Context])(postNorm)

  def applyWithHiddenStates(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): (List[Tensor2[Context, Embedding, V]], Tensor2[Context, Embedding, V]) =
    val allStates = layers.scanLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)
    val hiddenStates = allStates.tail // drop initial context
    val res = hiddenStates.last
    (hiddenStates, res.vmap(Axis[Context])(postNorm))

object CrossTransformer:

  def apply[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
      params: CrossTransformer.Params[CrossEmbedding, Embedding, V],
      createCrossAttentionMask: Shape2[Context, CrossContext] => Tensor2[Context, CrossContext, Bool],
      createSelfAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
  ) = new CrossTransformer(params, createCrossAttentionMask, createSelfAttentionMask)

  def apply[CrossContext: Λ, Context: Λ, CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](
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
      postNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[CrossEmbedding, Embedding, V] =
      new Params(
        transformerLayers =
          key.split(numTransformerLayers).map: key =>
            CrossTransformerLayer.Params.xavierUniformDepthScaled(numTransformerLayers, numHeads, crossEmbeddingExtent, embeddingExtent, embeddingMixedExtent, vtype, key)
          .toList,
        postNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
