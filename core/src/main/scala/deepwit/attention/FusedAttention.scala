package deepwit.attention

import dimwit.*
import dimwit.hardware.DeviceBackend
import dimwit.jax.Jax
import dimwit.python.PyBridge
import dimwit.Label as Λ
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** The fused cuDNN attention kernel, and what it takes to run it.
  *
  * `jax.nn.dot_product_attention` with `implementation = "cudnn"` attends without ever materializing
  * the scores, which is what makes it faster — measured at roughly 2x for a context of 1024 and 3x
  * at 4096, whether or not the attention is causal. Causal masking adds to that, because the kernel
  * skips the masked blocks rather than computing and discarding them.
  *
  * The kernel takes half precision only, which is why everything here is fixed to [[BFloat16]]
  * rather than generic: it rejects Float32 with a `NotImplementedError` rather than falling back. It
  * further needs a CUDA device — see [[isAvailable]] — and a head dimension that is a multiple of 8
  * and at most 128 (256 on Hopper and later), which JAX checks, naming the offending dimension.
  */
object FusedAttention:

  /** Whether cuDNN can run this element type here: half precision, on a CUDA device. */
  def isAvailable(dtype: DType): Boolean = dtype == DType.BFloat16 && hasCudaDevice

  /** The attended values per head, as the kernel returns them.
    *
    * @param isCausal Whether a target position may only attend to source positions up to its own index.
    */
  private[attention] def headValues[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16],
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16],
      isCausal: Boolean
  ): Tensor3[Head, Target, HeadValue, BFloat16] =
    headFirst(attend(params, source, target, isCausal).attended)

  /** The same, with the projections the kernel attended from.
    *
    * The kernel needs those projections anyway, so reporting them costs only their transposes —
    * which is the reason [[MultiHeadAttention.Intermediates]] stops short of the attention weights.
    */
  private[attention] def headValuesWithIntermediates[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16],
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16],
      isCausal: Boolean
  ): (Tensor3[Head, Target, HeadValue, BFloat16], MultiHeadAttention.Intermediates[Source, Target, BFloat16]) =
    val attention = attend(params, source, target, isCausal)
    (
      headFirst(attention.attended),
      (
        queries = headFirst(attention.queries),
        keys = headFirst(attention.keys),
        values = headFirst(attention.values)
      )
    )

  /** Everything the kernel touches, in the sequence-first layout it works in. */
  private def attend[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16],
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16],
      isCausal: Boolean
  ): (
      attended: Tensor3[Target, Head, HeadValue, BFloat16],
      queries: Tensor3[Target, Head, HeadQuery, BFloat16],
      keys: Tensor3[Source, Head, HeadKey, BFloat16],
      values: Tensor3[Source, Head, HeadValue, BFloat16]
  ) =
    // Projecting every head at once already gives the (sequence, head, head space) layout the kernel
    // wants; only the batch axis this attention does not have has to be faked and dropped again.
    val queries = target.dot(Axis[TargetEmbedding])(params.queryWeights)
    val keys = source.dot(Axis[SourceEmbedding])(params.keyWeights)
    val values = source.dot(Axis[SourceEmbedding])(params.valueWeights)

    // The kernel scales by one over the square root of the query space, as `ScaledDotProduct` does.
    val attended = jaxNn.dot_product_attention(
      batched(queries),
      batched(keys),
      batched(values),
      is_causal = isCausal,
      implementation = "cudnn"
    )

    (
      attended = PyBridge.liftPyTensor[(Target, Head, HeadValue), BFloat16](Jax.jnp.squeeze(attended, 0)),
      queries = queries,
      keys = keys,
      values = values
    )

  /** The head-by-head formulation leads with the head, where the kernel leads with the sequence. */
  private def headFirst[Sequence: Λ, HeadSpace: Λ](
      tensor: Tensor3[Sequence, Head, HeadSpace, BFloat16]
  ): Tensor3[Head, Sequence, HeadSpace, BFloat16] =
    PyBridge.liftPyTensor[(Head, Sequence, HeadSpace), BFloat16](
      Jax.jnp.transpose(PyBridge.toPyTensor(tensor), Seq(1, 0, 2).toPythonCopy)
    )

  /** Whether these parameters can go through the kernel.
    *
    * The weights have to hold what their type claims: an initializer that answers a Float32 array to
    * a `VType[BFloat16]` request would send cuDNN a dtype it rejects outright.
    */
  private[attention] def canRun[SourceEmbedding, TargetEmbedding, V: IsFloating](
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, V]
  ): Boolean =
    summon[IsFloating[V]].dtype == params.queryWeights.dtype && isAvailable(params.queryWeights.dtype)

  private def batched[T <: Tuple](tensor: Tensor[T, BFloat16]): py.Dynamic =
    Jax.jnp.expand_dims(PyBridge.toPyTensor(tensor), 0)

  private lazy val jaxNn = py.module("jax.nn")

  private lazy val hasCudaDevice: Boolean =
    // Asking for a backend that is not there raises rather than answering.
    scala.util.Try(DeviceBackend.GPU.devices.nonEmpty).getOrElse(false)
