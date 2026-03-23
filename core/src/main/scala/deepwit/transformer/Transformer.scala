package deepwit.transformer

import dimwit.*
import deepwit.labels.{Head, HeadKey, HeadQuery, HeadValue}
import deepwit.normalization.LayerNorm

trait Transformer[Context: Label, Embedding: Label](
    hyperParams: Transformer.HyperParams[Context, Embedding]
)(
    params: Transformer.Params[Embedding]
) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  private val layers =
    // fix hyper parameter
    val transformerLayerF = this.transformerLayer(hyperParams.transformerLayer)
    // create layer for each parameter
    params.transformerLayers.map(transformerLayerF)
  private val postNorm = LayerNorm(hyperParams.postNorm)(params.postNorm)

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(context_i)
    res.vmap(Axis[Context])(postNorm)

  protected def transformerLayer(hyperParams: TransformerLayer.HyperParams[Context, Embedding])(params: TransformerLayer.Params[Embedding]): TransformerLayer[Context, Embedding]

case class CausalTransformer[Context: Label, Embedding: Label](
    hyperParams: Transformer.HyperParams[Context, Embedding]
)(
    params: Transformer.Params[Embedding]
) extends Transformer[Context, Embedding](hyperParams)(params):

  protected override def transformerLayer(hyperParams: TransformerLayer.HyperParams[Context, Embedding])(params: TransformerLayer.Params[Embedding]): TransformerLayer[Context, Embedding] =
    CausalTransformerLayer(hyperParams)(params)

case class BidirectionalTransformer[Context: Label, Embedding: Label](
    hyperParams: Transformer.HyperParams[Context, Embedding]
)(
    params: Transformer.Params[Embedding]
) extends Transformer[Context, Embedding](hyperParams)(params):

  protected override def transformerLayer(hyperParams: TransformerLayer.HyperParams[Context, Embedding])(params: TransformerLayer.Params[Embedding]): TransformerLayer[Context, Embedding] =
    BidirectionalTransformerLayer(hyperParams)(params)

object Transformer:

  case class HyperParams[Context, Embedding](
      transformerLayer: TransformerLayer.HyperParams[Context, Embedding],
      postNorm: LayerNorm.HyperParams
  )

  case class Params[Embedding](
      transformerLayers: List[TransformerLayer.Params[Embedding]],
      postNorm: LayerNorm.Params[Embedding]
  )

  object Params:

    def xavierUniformDepthScaled[E: Label](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[E], embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed], key: Random.Key): Params[E] =
      new Params[E](
        transformerLayers =
          key.split(numTransformerLayers).map: key =>
            TransformerLayer.Params.xavierUniformDepthScaled(numTransformerLayers)(headExtent, headQueryExtent, headKeyExtent, headValueExtent, embeddingExtent, embeddingMixedExtent, key)
          .toList,
        postNorm = LayerNorm.Params.identity(embeddingExtent)
      )
