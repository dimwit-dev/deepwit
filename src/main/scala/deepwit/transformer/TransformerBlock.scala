package deepwit.transformer

import dimwit.*

case class TransformerBlock[Context: Label, Embedding](layers: List[ITransformerLayer[Context, Embedding]]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
  override def apply(context: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
    layers.foldLeft(context):
      case (context_i, layer) => layer(context_i)
