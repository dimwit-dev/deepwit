package deepwit.transformer.attention

import dimwit.Label

/** Default axis labels for the query, key and value spaces of a single [[Attention]]. */
trait Query derives Label
trait Key derives Label
trait Value derives Label
trait AttentionWeights derives Label

/** Axis labels for the per-head spaces of a [[MultiHeadAttention]]. */
trait Head derives Label
trait HeadQuery derives Label
trait HeadKey derives Label
trait HeadValue derives Label
