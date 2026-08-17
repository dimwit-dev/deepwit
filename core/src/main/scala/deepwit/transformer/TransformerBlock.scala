package deepwit.transformer

import dimwit.*
import dimwit.Label as Λ

/** The residual skeleton of a transformer layer.
  *
  * The context is mixed along itself and then along the embedding, each on its own residual branch.
  * What each mixer is remains open to the implementation.
  *
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param contextAxis The axis of the sequence being mixed.
  */
trait TransformerBlock[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  override final def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val contextMixed = context + contextMixer(context)
    contextMixed + contextMixed.vmap(Axis[Context])(embeddingMixer)

  protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V]

  protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V]
