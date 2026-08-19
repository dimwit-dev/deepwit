package deepwit.attention

import dimwit.Label

/** Axis labels for the per-head spaces of a [[MultiHeadAttention]]. */
trait Head derives Label
trait HeadQuery derives Label
trait HeadKey derives Label
trait HeadValue derives Label
