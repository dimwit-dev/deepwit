package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm
import dimwit.Label as Λ

case class CrossTransformer[CrossContext: Λ, CrossEmbedding, Context: Λ, Embedding: Λ, V: IsFloating](
    layers: List[CrossTransformerLayer[CrossContext, CrossEmbedding, Context, Embedding, V]],
    postNorm: LayerNorm[Embedding, V]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)
    res.vmap(Axis[Context])(postNorm)
