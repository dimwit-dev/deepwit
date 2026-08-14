package deepwit.transformer.attention

import dimwit.{Key as _, *}
import dimwit.Conversions.given
import deepwit.base.softmax
import dimwit.stats.Normal
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.init
import dimwit.Label as Λ

import me.shadaj.scalapy.py
import dimwit.python.PyBridge
import deepwit.transformer.{causalMask, fullMask}
import deepwit.transformer.MLPEmbeddingMixer

class MultiHeadSelfAttention[Target: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V],
    createAttentionMask: Shape2[Target, Target] => Tensor2[Target, Target, Bool]
) extends ((Tensor2[Target, Embedding, V]) => Tensor2[Target, Embedding, V]):

  private val multiHeadAttention = MultiHeadAttention(MultiHeadAttention.Params(params.queryWeights, params.keyWeights, params.valueWeights, params.headWeights), createAttentionMask)

  override def apply(context: Tensor2[Target, Embedding, V]): Tensor2[Target, Embedding, V] = multiHeadAttention(context, context)

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
      headWeights: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      require(embeddingExtent.size % numHeads == 0)
      import MultiHeadAttention.Params.{xavierUniformHeads, xavierUniformHeadWeights}
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val headExtent = Axis[Head] -> numHeads
      val headQueryExtent = Axis[HeadQuery] -> embeddingExtent.size / numHeads
      val headKeyExtent = Axis[HeadKey] -> embeddingExtent.size / numHeads
      val headValueExtent = Axis[HeadValue] -> embeddingExtent.size / numHeads
      Params(
        queryWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headQueryExtent, vtype, queryKey),
        keyWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headKeyExtent, vtype, keyKey),
        valueWeights = xavierUniformHeads(headExtent.size, embeddingExtent, headValueExtent, vtype, valueKey),
        headWeights = xavierUniformHeadWeights(numTransformerLayers, headExtent * headValueExtent, embeddingExtent, vtype, projectionKey)
      )
