/** Contains the fused attention kernel and the multi-head attention classes that use it.
  * Fused attention kernel allows use of flash attention on CUDA devices with cuDNN.
  */

package deepwit.attention

import dimwit.*
import dimwit.hardware.DeviceBackend
import dimwit.jax.Jax
import dimwit.python.PyBridge
import dimwit.Label as Λ
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

private[attention] object FusedAttentionKernel:

  /** Whether cuDNN can run this element type here: half precision, on a CUDA device. */
  def canRun(dtype: DType): Boolean = dtype == DType.BFloat16 && hasCudaDevice

  /** The same, with the projections the kernel attended from.
    *
    * The kernel needs those projections anyway, so reporting them costs only their transposes —
    * which is the reason [[MultiHeadAttention.Intermediates]] stops short of the attention weights.
    */
  def attend[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
      params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16],
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16],
      isCausal: Boolean
  ): (Tensor3[Target, Head, HeadValue, BFloat16], MultiHeadAttention.Intermediates[Source, Target, BFloat16]) =
    val queries = target.dot(Axis[TargetEmbedding])(params.queryWeights)
    val keys = source.dot(Axis[SourceEmbedding])(params.keyWeights)
    val values = source.dot(Axis[SourceEmbedding])(params.valueWeights)

    val res = PyBridge.liftPyTensor[(Target, Head, HeadValue), BFloat16]:
      Jax.jax.nn.dot_product_attention(
        PyBridge.toPyTensor(queries),
        PyBridge.toPyTensor(keys),
        PyBridge.toPyTensor(values),
        is_causal = isCausal,
        implementation = "cudnn"
      )

    (
      res,
      (queries = queries, keys = keys, values = values)
    )

  private lazy val hasCudaDevice: Boolean = scala.util.Try(DeviceBackend.GPU.devices.nonEmpty).getOrElse(false)

trait MultiHeadFusedAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ]:
  this: MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, BFloat16] =>
  protected final def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, BFloat16]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, BFloat16] =
    throw new RuntimeException("Fused attention does not run head-by-head, so this method should never be called.")

/** Multi-head unrestricted attention as run by cuDNN's fused kernel.
  *
  * Fixed to [[BFloat16]] rather than generic in the element type, because that is the constraint the
  * kernel actually has — see [[FusedAttention]] for the rest of them. It computes what
  * [[MultiHeadFullAttention]] computes, and the two are required to agree.
  *
  * Note the head dimensions: cuDNN attends with the value space the key space has, so this only
  * covers parameters whose source and target embeddings are equally wide. Cross-attention between
  * differently sized embeddings stays on the head-by-head formulation.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
class MultiHeadFusedFullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]
) extends MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, BFloat16](sourceAxis, targetAxis, params) with MultiHeadFusedAttention[Source, SourceEmbedding, Target, TargetEmbedding]:

  override protected def attend(
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16]
  ): (Tensor3[Target, Head, HeadValue, BFloat16], MultiHeadAttention.Intermediates[Source, Target, BFloat16]) =
    FusedAttentionKernel.attend(params, source, target, isCausal = false)

/** Multi-head causal attention as run by cuDNN's fused kernel.
  *
  * Fixed to [[BFloat16]] rather than generic in the element type, because that is the constraint the
  * kernel actually has — see [[FusedAttention]] for the rest of them. It computes what
  * [[MultiHeadCausalAttention]] computes, and the two are required to agree.
  *
  * @param sourceAxis The axis of the source sequence. Names the label the source is attended over.
  * @param targetAxis The axis of the target sequence. Names the label the queries are drawn from.
  */
class MultiHeadFusedCausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]
) extends MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, BFloat16](sourceAxis, targetAxis, params) with MultiHeadFusedAttention[Source, SourceEmbedding, Target, TargetEmbedding]:

  override protected def attend(
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16]
  ): (Tensor3[Target, Head, HeadValue, BFloat16], MultiHeadAttention.Intermediates[Source, Target, BFloat16]) =
    FusedAttentionKernel.attend(params, source, target, isCausal = true)
