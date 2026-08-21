package deepwit.attention

import dimwit.*
import dimwit.hardware.DeviceBackend
import dimwit.jax.Jax
import dimwit.python.PyBridge
import dimwit.Label as Λ
import me.shadaj.scalapy.py
import scala.util.Try

/** Marker trait for multi-head attention implementations that use [[FusedMultiHeadAttentionKernel]]. Restricted to BFloat16 due to kernel constraints. */
trait MultiHeadFusedAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ]:
  this: MultiHeadAttention[Source, SourceEmbedding, Target, TargetEmbedding, BFloat16] =>
  protected final def headAttention(
      headParams: Attention.Params[SourceEmbedding, TargetEmbedding, HeadQuery, HeadKey, HeadValue, BFloat16]
  ): Attention[Source, SourceEmbedding, Target, TargetEmbedding, HeadQuery, HeadKey, HeadValue, BFloat16] =
    throw new RuntimeException("Fused attention does not run head-by-head, so this method should never be called.")

// --- Fused multi-head attention implementations ---

class MultiHeadFusedFullAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]
) extends MultiHeadFullAttention[Source, SourceEmbedding, Target, TargetEmbedding, BFloat16](sourceAxis, targetAxis, params) with MultiHeadFusedAttention[Source, SourceEmbedding, Target, TargetEmbedding]:

  override protected def attend(
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16]
  ): (Tensor3[Target, Head, HeadValue, BFloat16], MultiHeadAttention.Intermediates[Source, Target, BFloat16]) =
    FusedMultiHeadAttentionKernel.attend(params, source, target, isCausal = false)

object MultiHeadFusedFullAttention:
  export FusedMultiHeadAttentionKernel.canRun as isAvailable

class MultiHeadFusedCausalAttention[Source: Λ, SourceEmbedding: Λ, Target: Λ, TargetEmbedding: Λ](
    sourceAxis: Axis[Source],
    targetAxis: Axis[Target],
    params: MultiHeadAttention.Params[SourceEmbedding, TargetEmbedding, BFloat16]
) extends MultiHeadCausalAttention[Source, SourceEmbedding, Target, TargetEmbedding, BFloat16](sourceAxis, targetAxis, params) with MultiHeadFusedAttention[Source, SourceEmbedding, Target, TargetEmbedding]:

  override protected def attend(
      source: Tensor2[Source, SourceEmbedding, BFloat16],
      target: Tensor2[Target, TargetEmbedding, BFloat16]
  ): (Tensor3[Target, Head, HeadValue, BFloat16], MultiHeadAttention.Intermediates[Source, Target, BFloat16]) =
    FusedMultiHeadAttentionKernel.attend(params, source, target, isCausal = true)

object MultiHeadFusedCausalAttention:
  export FusedMultiHeadAttentionKernel.canRun as isAvailable

/** Fused multi-head attention kernel to gain more efficiency.
  *
  * This kernel is only available for BFloat16 on CUDA devices. It enables [FlashAttention](https://arxiv.org/abs/2205.14135) and other optimizations that are not available in the unfused implementation.
  */
private[attention] object FusedMultiHeadAttentionKernel:

  /** Whever fused kernel can run for given data type. */
  def canRun(dtype: DType): Boolean =
    val hasCudaDevice = Try(DeviceBackend.GPU.devices.nonEmpty).getOrElse(false)
    dtype == DType.BFloat16 && hasCudaDevice

  /** Kernel implementation. */
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
      Jax.jnn.dot_product_attention(
        PyBridge.toPyTensor(queries),
        PyBridge.toPyTensor(keys),
        PyBridge.toPyTensor(values),
        is_causal = isCausal,
        implementation = "cudnn"
      )

    (res, (queries = queries, keys = keys, values = values))
