package deepwit.examples.autoencoder

import dimwit.{Label, |*|}
import deepwit.examples.dataset.MNISTLoader

export MNISTLoader.{Height, Width}

type Pixel = Height |*| Width
type ReconstructedPixel = Pixel

trait EHidden1 derives Label
trait EHidden2 derives Label
trait Latent derives Label
trait DHidden1 derives Label
trait DHidden2 derives Label
