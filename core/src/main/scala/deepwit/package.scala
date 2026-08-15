package deepwit

import dimwit.*

/** The smallest representable difference for `dtype`, i.e. machine epsilon.
  *
  * Serves as the default stability constant wherever a computation would otherwise risk a
  * division by zero or a logarithm of zero.
  */
private[deepwit] def defaultEpsilon(dtype: DType): Float = dimwit.jax.Jax.jnp.finfo(dtype.jaxType).eps.as[Float]

/** Resolves an epsilon that is either given as a fixed value or derived from the scalar type. */
private[deepwit] def unwrapEpsilon(eps: Float | (DType => Float), dtype: DType): Float = eps match
  case ε: Float            => ε
  case f: (DType => Float) => f(dtype)
