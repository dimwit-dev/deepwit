package deepwit.attention

import dimwit.*

/** The multi-head attention heads. */
trait Head derives Label

/** The space a head projects queries into. */
trait HeadQuery derives Label

/** The space a head projects keys into. */
trait HeadKey derives Label

/** The space a head projects values into. */
trait HeadValue derives Label

/** Which source positions each target position may attend to, given the positions of both:
  * `mask(targetPositions, sourcePositions)(row, col) == true` allows the target at `row` to attend to the source at `col`.
  */
type AttentionMask[Target, Source] = (Tensor1[Target, Int32], Tensor1[Source, Int32]) => Tensor2[Target, Source, Bool]
