package deepwit

/** Global test utility definitions */

import dimwit.*


import org.scalatest.matchers.{Matcher, MatchResult}

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label
trait E derives Label

def approxEqual[T <: Tuple: Labels, V](
    right: Tensor[T, V],
    tolerance: Float = 1e-6f
)(using ev: IsFloating[V]): Matcher[Tensor[T, V]] =
  new Matcher[Tensor[T, V]]:
    def apply(left: Tensor[T, V]): MatchResult =
      val leftF = left.asInstanceOf[Tensor[T, Float32]]
      val rightF = right.asInstanceOf[Tensor[T, Float32]]

      val areEqual = (leftF `approxEquals` (rightF, tolerance)).item
      lazy val diffMsg =
        if areEqual then "" else s"Max diff: ${(leftF - rightF).abs.max}"

      MatchResult(
        areEqual,
        s"Tensors did not match ($diffMsg).\nLeft: $left\nRight: $right",
        s"Tensors matched, but they shouldn't have."
      )
