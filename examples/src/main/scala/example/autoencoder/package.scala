package example

import dimwit.{Label, |*|}
import examples.dataset.MNISTLoader

package object autoencoder:
  export MNISTLoader.{Height, Width}

  trait Hidden derives Label
  trait Output derives Label

  type Pixel = Height |*| Width
  type ReconstructedPixel = Height |*| Width

  trait EHidden1 derives Label
  trait EHidden2 derives Label
  trait Latent derives Label
  trait DHidden1 derives Label
  trait DHidden2 derives Label
