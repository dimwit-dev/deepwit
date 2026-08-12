import dimwit.*
import dimwit.Conversions.given
import deepwit.logging.TensorTreeLogger

import deepwit.example.diffusion.*

import examples.dataset.MNISTLoader
import MNISTLoader.TestSample

import plotwit.plotting.Plotting
import plotwit.plotting.Plotting.{hconcat, grid}

import viz.PlotTargets.websocket
import dimwit.stats.Normal
import MNISTLoader.{Width, Height}

dimwit.initialize()

def toUInt8Img(img2d: Tensor2[Height, Width, Float32]): Tensor2[Height, Width, UInt8] =
  val clipped = img2d.clip(-1f, 1f)
  ((clipped +! 1f) /! 2.0f *! 255.0f).asInt(VType[UInt8])

val logger = new TensorTreeLogger("/Users/mebr/Documents/Privat/Projects/deepwit/examples/out/DiffusionMNIST/20260810_231115")
val state = logger.load[TrainState](9_500).get
val hyperParams = DiffusionUNet.HyperParams.default
val model = DiffusionUNet(hyperParams)(state.params)

val testDataset = MNISTLoader.createTestDataset().get
val original = testDataset.images.slice(Axis[TestSample].at(0 until 64))

val batchedImage = original *! 2.0f -! 1.0f

val t = Tensor0(0.5f)
val alpha = (1f - t).sqrt
val sigma = t.sqrt
val noiseKey = Random.Key(1234)

val noise = Normal.standardIsotropic(batchedImage.shape, sigma).sample(noiseKey)
val noisyImage = batchedImage *! alpha + noise

val predictedNoise = noisyImage.vmap(Axis[TestSample])(img => model(img, t))
val reconstructed = (noisyImage - predictedNoise *! sigma) /! alpha

val specs = batchedImage.unstack(Axis[TestSample])
  .zip(noisyImage.unstack(Axis[TestSample]))
  .zip(reconstructed.unstack(Axis[TestSample]))
  .map:
    case ((original, noisy), reconstruction) =>
      hconcat(Seq(
        Plotting.image.plot(toUInt8Img(original).transpose, _.mark.width := 56, _.mark.height := 56, _.title := ""),
        Plotting.image.plot(toUInt8Img(noisy).transpose, _.mark.width := 56, _.mark.height := 56, _.title := ""),
        Plotting.image.plot(toUInt8Img(reconstruction).transpose, _.mark.width := 56, _.mark.height := 56, _.title := "")
      ))

Plotting.display(grid(specs.grouped(8).toSeq))

def generate() =
  val genKey = Random.Key(42)
  val pureNoise = Normal.standardNormal(batchedImage.shape).sample(genKey)

  val numSteps = 50
  val times = (0 to numSteps).map(i => 0.999f * (1f - i.toFloat / numSteps))

  val generatedImages = times.zip(times.tail).foldLeft(pureNoise):
    case (xt, (tCurr, tNext)) =>
      val t = Tensor0(tCurr)
      val tN = Tensor0(tNext)

      val alpha = (1f - t).sqrt
      val sigma = t.sqrt
      val alphaNext = (1f - tN).sqrt
      val sigmaNext = tN.sqrt

      println((tCurr, tNext))

      // 1. Predict noise
      val predNoise = xt.vmap(Axis[TestSample])(img => model(img, t))

      // 2. Estimate unclipped x0
      val x0Hat = (xt - predNoise *! sigma) /! alpha

      // 3. Step forward using the pure mathematical formula
      x0Hat *! alphaNext + predNoise *! sigmaNext

  val genSpecs = generatedImages.unstack(Axis[TestSample]).map: img =>
    Plotting.image.plot(
      toUInt8Img(img).transpose,
      _.mark.width := 56,
      _.mark.height := 56,
      _.title := ""
    )

  import viz.PlotTargets.desktopBrowser
  Plotting.display(grid(genSpecs.grouped(8).toSeq))
