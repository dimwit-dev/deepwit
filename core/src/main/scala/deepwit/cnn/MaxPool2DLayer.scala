package deepwit.cnn

import dimwit.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.{liftPyTensor, toPyTensor}
import dimwit.Label as Λ

class MaxPool2DLayer[S1: Λ, S2: Λ, V: IsFloating](
    window: Window2[S1, S2] | Int,
    stride: Stride2[S1, S2] | Int = 1,
    padding: Padding = Padding.SAME
) extends (Tensor2[S1, S2, V] => Tensor2[S1, S2, V]):

  override def apply(x: Tensor2[S1, S2, V]): Tensor2[S1, S2, V] =
    val winDims = window match
      case w: Int                                 => (w, w)
      case (w1: AxisExtent[?], w2: AxisExtent[?]) => (w1.size, w2.size)

    val winStrides = stride match
      case s: Int                                 => (s, s)
      case (s1: AxisExtent[?], s2: AxisExtent[?]) => (s1.size, s2.size)

    val padMode = padding match
      case Padding.SAME  => "SAME"
      case Padding.VALID => "VALID"

    liftPyTensor(
      Jax.lax.reduce_window(
        toPyTensor(x),
        Float.NegativeInfinity,
        Jax.lax.max,
        winDims,
        winStrides,
        padMode
      )
    )
