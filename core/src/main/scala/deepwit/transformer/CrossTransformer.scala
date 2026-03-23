package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm

case class CrossTransformer[CrossContext: Label, CrossEmbedding, Context: Label, Embedding: Label](
    layers: List[CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding]],
    postNorm: LayerNorm[Embedding]
) extends ((Tensor2[CrossContext, CrossEmbedding, Float], Tensor2[Context, Embedding, Float]) => Tensor2[Context, Embedding, Float]):

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, Float], context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)
    res.vmap(Axis[Context])(postNorm)
