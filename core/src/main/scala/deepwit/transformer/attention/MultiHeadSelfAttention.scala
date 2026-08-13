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

class MultiHeadSelfAttention[Context: Λ, Embedding: Λ, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V],
    createAttentionMask: Shape2[Context, Context] => Tensor2[Context, Context, Bool]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val headProjectionLayer = AffineLayer(params.headProjection)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val heads = headAttention(context)
    heads.vmap(Axis[Context])(heads => headProjectionLayer(heads.flatten))

  private def headAttention(context: Tensor2[Context, Embedding, V]): Tensor[(Head, Context, HeadValue), V] =
    zipvmap(Axis[Head])(params.wq, params.wk, params.wv):
      case (wq, wk, wv) =>
        val selfAttentionHead = SelfAttention(SelfAttention.Params(wq, wk, wv), createAttentionMask)
        selfAttentionHead(context)

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
      wq: Tensor3[Head, Embedding, HeadQuery, V],
      wk: Tensor3[Head, Embedding, HeadKey, V],
      wv: Tensor3[Head, Embedding, HeadValue, V],
      headProjection: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val nHeads = headExtent.size
      val headProjectionGain = Math.sqrt(1.0 / (2 * numTransformerLayers)).toFloat
      Params(
        wq = stack(queryKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headQueryExtent, vtype, key)), Axis[Head]),
        wk = stack(keyKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headKeyExtent, vtype, key)), Axis[Head]),
        wv = stack(valueKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headValueExtent, vtype, key)), Axis[Head]),
        headProjection = AffineLayer.Params.xavierUniform(headExtent * headValueExtent, embeddingExtent, vtype, projectionKey, gain = headProjectionGain)
      )
