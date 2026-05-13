package deepwit.transformer

import dimwit.*
import deepwit.labels.{Head, HeadKey, HeadQuery, HeadValue}
import deepwit.normalization.LayerNorm

trait Transformer[Context: Label, Embedding: Label, V: IsFloating](
    hyperParams: Transformer.HyperParams[Context, Embedding]
)(
    params: Transformer.Params[Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val layers = params.transformerLayers.map(this.transformerLayer)
  private val postNorm = LayerNorm(hyperParams.postNorm)(params.postNorm)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(context_i)
    res.vmap(Axis[Context])(postNorm)

  protected def transformerLayer(params: TransformerLayer.Params[Embedding, V]): TransformerLayer[Context, Embedding, V]

case class CausalTransformer[Context: Label, Embedding: Label, V: IsFloating](
    hyperParams: Transformer.HyperParams[Context, Embedding]
)(
    params: Transformer.Params[Embedding, V]
) extends Transformer[Context, Embedding, V](hyperParams)(params):

  protected override def transformerLayer(params: TransformerLayer.Params[Embedding, V]): TransformerLayer[Context, Embedding, V] =
    CausalTransformerLayer(params)

case class BidirectionalTransformer[Context: Label, Embedding: Label, V: IsFloating](
    hyperParams: Transformer.HyperParams[Context, Embedding]
)(
    params: Transformer.Params[Embedding, V]
) extends Transformer[Context, Embedding, V](hyperParams)(params):

  protected override def transformerLayer(params: TransformerLayer.Params[Embedding, V]): TransformerLayer[Context, Embedding, V] =
    BidirectionalTransformerLayer(params)

object Transformer:

  case class HyperParams[Context, Embedding](
      postNorm: LayerNorm.HyperParams
  )

  case class Params[Embedding, V](
      transformerLayers: List[TransformerLayer.Params[Embedding, V]],
      postNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Label, V: IsFloating](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      new Params[Embedding, V](
        transformerLayers =
          key.split(numTransformerLayers).map: key =>
            TransformerLayer.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, embeddingMixedExtent, vtype, key)
          .toList,
        postNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
