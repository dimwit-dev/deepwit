package deepwit.examples.variationalAutoencoder

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.{Adam, AdamState}

import deepwit.examples.dataset.MNISTLoader

import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.loss.BinaryCrossEntropy
import deepwit.training.{Monitor, tapEvery}

case class TrainState(
    params: VariationalAutoencoder.Params,
    optimizerState: AdamState[VariationalAutoencoder.Params],
    trainKey: Key,
    lastCost: Tensor0[Float32]
)

@main
def train(): Unit =

  // -- Configuration --

  val batchSize = 256
  val numIterations = 10_000
  val latentDim = 20
  val learningRate = 3e-4f

  val eHidden1Extent = Axis[EHidden1] -> 512
  val eHidden2Extent = Axis[EHidden2] -> 256
  val latentExtent = Axis[Latent] -> latentDim
  val dHidden1Extent = Axis[DHidden1] -> 256
  val dHidden2Extent = Axis[DHidden2] -> 512

  // -- Prepare training data --

  trait Batch derives Label

  val (initKey, samplingSeed, testKey) = Key(42).splitToTuple(3)

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val testDataset = MNISTLoader.createTestDataset().get

  val trainDataStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize).map(_.images)

  // -- Prepare train trajectory --

  val optimizer = Adam(learningRate = learningRate)

  def negativeElboFor[S: Label](samples: Tensor3[S, Height, Width, Float32], key: Key)(params: VariationalAutoencoder.Params): Tensor0[Float32] =
    val model = VariationalAutoencoder(params)
    val sampleKeys = key.splitToTensor(Axis[S] -> samples.shape(Axis[S]))
    zipvmap(Axis[S])(samples, sampleKeys): (sample, sampleKey) =>
      val original = sample.flatten
      val posterior = model.posterior(original)
      val latent = posterior.sample(sampleKey.item)
      val reconstructionCost =
        zipvmap(Axis[Pixel])(original, model.decoder.logits(latent)): (origPixel, reconLogit) =>
          BinaryCrossEntropy.fromLogits(origPixel, reconLogit)
        .sum
      val klCost =
        val mean = posterior.loc
        val variance = posterior.scale * posterior.scale
        0.5f * ((mean * mean + variance - variance.log) -! 1f).sum
      reconstructionCost + klCost
    .mean

  def gradientStep(batch: Tensor3[Batch, Height, Width, Float32], state: TrainState): TrainState =
    val (nextKey, stepKey) = state.trainKey.split2()
    val (cost, grads) = Autodiff.valueAndGrad(negativeElboFor(batch, stepKey))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, nextKey, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialState =
    val initialParams = VariationalAutoencoder.Params.init(eHidden1Extent, eHidden2Extent, latentExtent, dHidden1Extent, dHidden2Extent, initKey)
    val initialOptimizerState = optimizer.init(initialParams)
    TrainState(initialParams, initialOptimizerState, samplingSeed, Tensor0(-1f))

  val trainTrajectory = trainDataStream.scanLeft(initialState):
    case (state, batch) =>
      jitGradientStep(batch, state)

  // -- Run train trajectory --

  val checkpointer = TensorTreeCheckpointer.newIn("out/VariationalAutoencoder")
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)
  val finalState = trainTrajectory
    .tapEvery(10):
      // Report training loss
      case (state, step) =>
        println(trainMonitor.report(step, state))
    .tapEvery(500):
      // Evaluate on the test set, from a key of its own so the estimate does not move with training
      case (state, step) =>
        val lossValue = negativeElboFor(testDataset.images, testKey)(state.params).item
        println(s"Step $step | Test loss: $lossValue")
    .tapEvery(500):
      // Save checkpoints
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Checkpoint saved at step $step")
    .drop(numIterations)
    .next()

  println(f"Final cost: ${finalState.lastCost.item}%.6f")
  println(s"Done. Wrote ${checkpointer.rootPath}.")
