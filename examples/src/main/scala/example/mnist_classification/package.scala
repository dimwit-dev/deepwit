package deepwit.example

import dimwit.{Label, |*|}
import deepwit.examples.dataset.MNISTLoader

package object mnist_classification:

  export MNISTLoader.{TrainSample, TestSample, Height, Width}

  trait Channel derives Label
  trait Hidden derives Label
  trait PixelEmbedding derives Label
  type ImageEmbedding = Height |*| Width |*| PixelEmbedding
  trait Output derives Label
