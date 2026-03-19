package deepwit.transformer

import dimwit.*
import deepwit.normalization.LayerNorm

case class Transformer[Context: Label, Embedding: Label](
    layers: List[TransformerLayer[Context, Embedding]],
    postNorm: LayerNorm[Embedding]
) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    val res = layers.foldLeft(context):
      case (context_i, layer) => layer(context_i)
    res.vmap(Axis[Context])(postNorm)
