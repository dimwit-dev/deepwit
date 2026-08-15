package deepwit.examples.mnistClassification

import dimwit.{Label, |*|}
import deepwit.examples.dataset.MNISTLoader

export MNISTLoader.{TrainSample, TestSample, Height, Width}

trait Channel derives Label
trait Hidden derives Label
trait PixelEmbedding derives Label
type ImageEmbedding = Height |*| Width |*| PixelEmbedding
trait Output derives Label
