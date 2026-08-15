package deepwit.transformer.attention

import dimwit.*
import deepwit.base.AffineLayer
import dimwit.Label as Λ

/** Represents multi-head self-attention, i.e. multi-head attention of a sequence onto itself.
  *
  * @tparam Target The axis label for the sequence.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param createAttentionMask A function generating a boolean mask to prevent attention to certain positions.
  */
class MultiHeadSelfAttention[Target: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V],
    createAttentionMask: Shape2[Target, Target] => Tensor2[Target, Target, Bool]
) extends (Tensor2[Target, Embedding, V] => Tensor2[Target, Embedding, V]):

  private val multiHeadAttention = MultiHeadAttention(
    MultiHeadAttention.Params(params.queryWeights, params.keyWeights, params.valueWeights, params.outputProjection),
    createAttentionMask
  )

  override def apply(target: Tensor2[Target, Embedding, V]): Tensor2[Target, Embedding, V] = multiHeadAttention(target, target)

/*
TODO reimplement this in new API
class MultiHeadCausalSelfFlashAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    axis: Axis[Context],
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val headProjectionLayer = AffineLayer(params.headProjection)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    /** Overwrite with jax.nn.dot_product_attention to enable FlashAttention on CUDA. Roughtly 1.5x faster */
    val queries = context.dot(Axis[Embedding])(params.wq)
    val keys = context.dot(Axis[Embedding])(params.wk)
    val values = context.dot(Axis[Embedding])(params.wv)

    val queriesJax = PyBridge.toPyTensor(queries)
    val keysJax = PyBridge.toPyTensor(keys)
    val valuesJax = PyBridge.toPyTensor(values)

    lazy val jaxNn = py.module("jax.nn")

    val resJax = jaxNn.dot_product_attention(
      queriesJax,
      keysJax,
      valuesJax,
      is_causal = true,
      implementation = "cudnn"
    )

    val attended = PyBridge.liftPyTensor(resJax).asInstanceOf[Tensor3[Context, Head, HeadValue, V]]
    attended.vmap(Axis[Context])(headsForContext => headProjectionLayer(headsForContext.flatten))*/

object MultiHeadSelfAttention:

  case class Params[Embedding, V](
      queryWeights: Tensor3[Head, Embedding, HeadQuery, V],
      keyWeights: Tensor3[Head, Embedding, HeadKey, V],
      valueWeights: Tensor3[Head, Embedding, HeadValue, V],
      outputProjection: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Key): Params[Embedding, V] =
      require(embeddingExtent.size % numHeads == 0)
      import MultiHeadAttention.Params.{xavierUniformHeads, xavierUniformOutputProjection}
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val headExtent = Axis[Head] -> numHeads
      val headQueryExtent = Axis[HeadQuery] -> embeddingExtent.size / numHeads
      val headKeyExtent = Axis[HeadKey] -> embeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> embeddingExtent.size / numHeads
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headQueryExtent, vtype, queryKey),
        keyWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headKeyExtent, vtype, keyKey),
        valueWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headValueExtent, vtype, valueKey),
        outputProjection = xavierUniformOutputProjection(numTransformerLayers, headExtent * headValueExtent, embeddingExtent, vtype, projectionKey)
      )
