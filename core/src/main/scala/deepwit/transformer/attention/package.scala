package deepwit.transformer

package object attention:

  import dimwit.Label

  trait Query derives Label
  trait Key derives Label
  trait Value derives Label
  trait AttentionWeights derives Label

  trait Head derives Label
  trait HeadQuery derives Label
  trait HeadKey derives Label
  trait HeadValue derives Label
