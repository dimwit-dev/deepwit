package deepwit.transformer

import dimwit.*
import dimwit.Label as Λ

/** The residual skeleton of a transformer block that additionally attends onto a cross context.
  *
  * The context is mixed along itself, then along the cross context, and finally along the embedding,
  * each on its own residual branch. What each mixer is remains open to the implementation.
  *
  * @tparam CrossContext The axis label for the cross sequence.
  * @tparam CrossEmbedding The axis label for the cross embedding space.
  * @tparam Context The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param crossContextAxis The axis of the sequence being attended onto.
  * @param contextAxis The axis of the sequence being mixed.
  */
trait CrossTransformerBlock[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    crossContextAxis: Axis[CrossContext],
    contextAxis: Axis[Context]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  override final def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val contextMixed = context + contextMixer(context)
    val crossContextMixed = contextMixed + crossContextMixer(crossContext, contextMixed)
    crossContextMixed + crossContextMixed.vmap(Axis[Context])(embeddingMixer)

  protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V]

  protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V]

  protected def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V]
