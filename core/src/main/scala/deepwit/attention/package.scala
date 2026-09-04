package deepwit.attention

import dimwit.Label

/** The multi-head attention heads. */
trait Head derives Label

/** The space a head projects queries into. */
trait HeadQuery derives Label

/** The space a head projects keys into. */
trait HeadKey derives Label

/** The space a head projects values into. */
trait HeadValue derives Label
