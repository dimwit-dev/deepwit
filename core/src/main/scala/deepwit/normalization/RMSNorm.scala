package deepwit.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax

case class RMSNorm[L: Label, V: IsFloating](
    hyperParams: RMSNorm.HyperParams
)(
    params: RMSNorm.Params[L, V]
) extends (Tensor1[L, V] => Tensor1[L, V]):

  private val epsilon = hyperParams.epsilon
  private def rescale(x: Tensor1[L, V]): Tensor1[L, V] =
    val variance = (x -! x.mean).pow(2).mean
    x /! (variance + epsilon).sqrt

  def apply(x: Tensor1[L, V]): Tensor1[L, V] =
    rescale(x) * params.weight

object RMSNorm:

  case class HyperParams(epsilon: Float)

  def apply[L: Label, V: IsFloating](params: Params[L, V]): RMSNorm[L, V] =
    val epsilon = Jax.jnp.finfo(params.weight.dtype.jaxType).eps.as[Float]
    RMSNorm(HyperParams(epsilon))(params)

  case class Params[L, V](weight: Tensor1[L, V])

  object Params:

    def identity[L: Label, V: IsFloating](ae: AxisExtent[L], vtype: VType[V]) =
      Params(weight = Tensor(Shape(ae), vtype).fill(1f))
