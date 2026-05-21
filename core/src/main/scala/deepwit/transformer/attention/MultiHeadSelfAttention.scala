package deepwit.transformer.attention

import dimwit.*
import dimwit.Conversions.given
import deepwit.base.ActivationFunction.softmax
import dimwit.stats.Normal
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.init

import me.shadaj.scalapy.py
import dimwit.python.PyBridge

trait MultiHeadSelfAttention[Context: Label, Embedding: Label, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val headProjectionLayer = AffineLayer(params.headProjection)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val heads = headAttention(context)
    heads.vmap(Axis[Context])(heads => headProjection(heads.flatten))

  protected def headAttention(context: Tensor2[Context, Embedding, V]): Tensor[(Head, Context, HeadValue), V] =
    zipvmap(Axis[Head])(params.wq, params.wk, params.wv):
      case (wq, wk, wv) =>
        selfAttention(SelfAttention.Params(wq, wk, wv))(context)

  protected def headProjection(headValues: Tensor1[Head |*| HeadValue, V]) = headProjectionLayer(headValues)

  protected def selfAttention(params: SelfAttention.Params[Embedding, HeadQuery, HeadKey, HeadValue, V]): SelfAttention[Context, Embedding, HeadQuery, HeadKey, HeadValue, V]

case class MultiHeadCausalSelfAttention[Context: Label, Embedding: Label, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](params):

  protected override def selfAttention(params: SelfAttention.Params[Embedding, HeadQuery, HeadKey, HeadValue, V]) = CausalSelfAttention(params)

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
    attended.vmap(Axis[Context])(headsForContext => headProjection(headsForContext.flatten))

case class MultiHeadFullSelfAttention[Context: Label, Embedding: Label, V: IsFloating](
    params: MultiHeadSelfAttention.Params[Embedding, V]
) extends MultiHeadSelfAttention[Context, Embedding, V](params):

  protected override def selfAttention(params: SelfAttention.Params[Embedding, HeadQuery, HeadKey, HeadValue, V]) = FullSelfAttention(params)

object MultiHeadSelfAttention:

  case class Params[Embedding, V](
      wq: Tensor3[Head, Embedding, HeadQuery, V],
      wk: Tensor3[Head, Embedding, HeadKey, V],
      wv: Tensor3[Head, Embedding, HeadValue, V],
      headProjection: AffineLayer.Params[Head |*| HeadValue, Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Label, V: IsFloating](numTransformerLayers: Int)(headExtent: AxisExtent[Head], headQueryExtent: AxisExtent[HeadQuery], headKeyExtent: AxisExtent[HeadKey], headValueExtent: AxisExtent[HeadValue], embeddingExtent: AxisExtent[Embedding], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (queryKey, keyKey, valueKey, projectionKey) = key.splitToTuple(4)
      val nHeads = headExtent.size
      val headProjectionGain = Math.sqrt(1.0 / (2 * numTransformerLayers)).toFloat
      Params(
        wq = stack(queryKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headQueryExtent, vtype, key)), Axis[Head]),
        wk = stack(keyKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headKeyExtent, vtype, key)), Axis[Head]),
        wv = stack(valueKey.split(nHeads).map(key => init.xavierUniform(embeddingExtent, headValueExtent, vtype, key)), Axis[Head]),
        headProjection = AffineLayer.Params.xavierUniform(headExtent * headValueExtent, embeddingExtent, vtype, projectionKey, gain = headProjectionGain)
      )
