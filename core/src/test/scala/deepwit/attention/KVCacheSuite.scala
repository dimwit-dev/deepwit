package deepwit.attention

import deepwit.*
import dimwit.*
import dimwit.stats.Normal
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class KVCacheSuite extends AnyFunSpec with Matchers:

  trait Ctx derives Label
  trait Emb derives Label

  private val ctxExtent = Axis[Ctx] -> 6
  private val params = MultiHeadSelfAttention.Params.init(numTransformerLayers = 2, numHeads = 2, embeddingExtent = Axis[Emb] -> 4, key = Random.Key(42))
  private val attention = MultiHeadCausalSelfAttention(Axis[Ctx], params)
  private val context = Normal.standardNormal(Shape(ctxExtent, Axis[Emb] -> 4)).sample(Random.Key(7))
  private val positions = Tensor(Shape1(ctxExtent)).fromArray(Array.range(0, ctxExtent.size))

  /** Decodes every position of `context`, one after another, starting from `cache`. */
  private def decodeFrom(cache: KVCache[Ctx, Float32]): Tensor2[Ctx, Emb, Float32] =
    val (_, attended) = scan(Axis[Ctx])(cache, (positions, context)):
      case (cache, (position, embedding)) =>
        val (attended, nextCache) = attention.decode(cache, position, embedding)
        (nextCache, attended)
    attended

  describe("decode"):

    it("reproduces causal attention over the whole context, one position at a time"):
      decodeFrom(KVCache.empty(ctxExtent, params)) should approxEqual(attention(context), 1e-5f)

    it("works under jit"):
      val decode = jit(attention.decode)
      val (attended, _) = decode(KVCache.empty(ctxExtent, params), Tensor0(0), context(Axis[Ctx].at(0)))
      attended should approxEqual(attention(context).select(Axis[Ctx].at(0)), 1e-5f)

  describe("prefill"):

    it("attends like apply and caches the keys and values of every position"):
      val (attended, cache) = attention.prefill(context)
      attended should approxEqual(attention(context), 1e-6f)
      decodeFrom(cache) should approxEqual(attention(context), 1e-5f)

    it("lets decoding continue after a prompt, whatever follows the prompt in the context"):
      val promptLength = 3
      val noise = Normal.standardNormal(context.shape).sample(Random.Key(8))
      val promptOnly = where_!(Tensor0(promptLength) >! positions, context, noise)
      val (_, cache) = attention.prefill(promptOnly)
      (promptLength until ctxExtent.size).foldLeft(cache): (cache, position) =>
        val (attended, nextCache) = attention.decode(cache, Tensor0(position), context(Axis[Ctx].at(position)))
        attended should approxEqual(attention(context).select(Axis[Ctx].at(position)), 1e-5f)
        nextCache
