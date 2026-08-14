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

  def applyWithHiddenStates(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): (List[Tensor2[Context, Embedding, V]], Tensor2[Context, Embedding, V]) =
    val allStates = layers.scanLeft(context):
      case (context_i, layer) => layer(crossContext, context_i)
    val hiddenStates = allStates.tail // drop initial context
    val res = hiddenStates.last
    (hiddenStates, res.vmap(Axis[Context])(postNorm))
