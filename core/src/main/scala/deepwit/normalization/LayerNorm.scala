package deepwit.normalization

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax

case class LayerNorm[L: Label, V: IsFloating](
    hyperParams: LayerNorm.HyperParams
)(
    params: LayerNorm.Params[L, V]
) extends (Tensor1[L, V] => Tensor1[L, V]):

  private val epsilon = hyperParams.epsilon
  private def standardize(x: Tensor1[L, V]): Tensor1[L, V] =
    val x0 = x -! x.mean
    val variance = x0.pow(2).mean
    x0 /! (variance + epsilon).sqrt

  def apply(x: Tensor1[L, V]): Tensor1[L, V] =
    standardize(x) * params.weight + params.bias

object LayerNorm:

  case class HyperParams(epsilon: Float)

  def apply[L: Label, V: IsFloating](params: Params[L, V]): LayerNorm[L, V] =
    val epsilon = Jax.jnp.finfo(params.weight.dtype.jaxType).eps.as[Float]
    LayerNorm(HyperParams(epsilon))(params)

  case class Params[L, V](weight: Tensor1[L, V], bias: Tensor1[L, V])

  object Params:
    def identity[L: Label, V: IsFloating](ae: AxisExtent[L], vtype: VType[V]) =
      Params(
        weight = Tensor(Shape(ae), vtype).fill(1f),
        bias = Tensor(Shape(ae), vtype).fill(0f)
      )
