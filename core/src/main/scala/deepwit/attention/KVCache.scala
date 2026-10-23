package deepwit.attention

import dimwit.*

/** The keys and values of every position of a context, so that decoding a position can attend to the
  * positions before it without projecting them again.
  *
  * The cache spans the whole context from the start, so its shape stays the same while decoding, as `jit`
  * and the loops of DimWit require. Positions not decoded yet hold arbitrary values, which are never attended to.
  *
  * @see [[MultiHeadCausalSelfAttention.prefill]] and [[MultiHeadCausalSelfAttention.decode]]
  */
case class KVCache[Context, V](
    keys: Tensor3[Context, Head, HeadKey, V],
    values: Tensor3[Context, Head, HeadValue, V]
)

object KVCache:

  /** A cache spanning `contextExtent` for the attention with `params`, before any position is decoded. */
  def empty[Context: Label, Embedding, V: IsFloating](contextExtent: AxisExtent[Context], params: MultiHeadSelfAttention.Params[Embedding, V]): KVCache[Context, V] =
    val keyWeights = params.multiHeadAttention.keyWeights
    val valueWeights = params.multiHeadAttention.valueWeights
    val headExtent = keyWeights.shape.extent(Axis[Head])
    KVCache(
      keys = Tensor(Shape(contextExtent, headExtent, keyWeights.shape.extent(Axis[HeadKey])), VType[V]).fill(0f),
      values = Tensor(Shape(contextExtent, headExtent, valueWeights.shape.extent(Axis[HeadValue])), VType[V]).fill(0f)
    )
