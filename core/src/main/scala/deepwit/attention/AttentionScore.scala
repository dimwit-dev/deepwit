package deepwit.attention

import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

/** Scores how strongly each target position attends to each source position, before masking.
  *
  * @tparam Target The axis label for the target sequence.
  * @tparam Source The axis label for the source sequence.
  * @tparam Query The axis label for the query space.
  * @tparam Key The axis label for the key space.
  * @tparam V The floating-point scalar type of the tensor elements.
  */
@FunctionalInterface
trait AttentionScore[Target, Source, Query, Key, V] extends ((Tensor2[Target, Query, V], Tensor2[Source, Key, V]) => Tensor2[Target, Source, V])

/** Scores a query against a key by their dot product, scaled down by the square root of the key extent.
  *
  * The scaling keeps the scores in a range where the softmax does not saturate as the key space grows.
  */
class ScaledDotProduct[Target: Λ, Source: Λ, Query: Λ, Key: Λ, V: IsFloating] extends AttentionScore[Target, Source, Query, Key, V]:

  override def apply(queries: Tensor2[Target, Query, V], keys: Tensor2[Source, Key, V]): Tensor2[Target, Source, V] =
    val dk = Math.sqrt(keys.shape(Axis[Key]))
    queries.dot(Axis[Query] -> Axis[Key])(keys) /! dk
