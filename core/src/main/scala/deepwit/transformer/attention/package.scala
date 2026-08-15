package deepwit.transformer

package object attention:

  import dimwit.Label

  trait Head derives Label
  trait HeadQuery derives Label
  trait HeadKey derives Label
  trait HeadValue derives Label
