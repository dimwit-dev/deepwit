package deepwit.examples.thinning

import dimwit.*

import deepwit.examples.dataset.TwoMoons
import deepwit.examples.dataset.TwoMoons.{Feature, Output}
import dimwit.Conversions.given
import dimwit.optimizer.{Adam, AdamState}

import deepwit.loss.CategoricalCrossEntropy
import deepwit.training.{Monitor, tapEvery}
import deepwit.checkpointing.TensorTreeCheckpointer

case class TrainState(
    params: MoonsMLP.Params,
    optimizerState: AdamState[MoonsMLP.Params],
    trainKey: Key,
    lastCost: Tensor0[Float32]
)

/** Fits the classifier to the two moons against a freshly thinned model at every step.
  *
  * Thinning appears exactly once, in [[gradientStep]]. The optimizer never sees it: it steps the
  * stored parameters, which are the same tree that was differentiated, so the checkpoint holds
  * nothing but weights. See the README in this directory.
  */
@main
def train(): Unit =

  // -- Configuration --

  val numIterations = 4_000
  val numSamples = 400
  val batchSize = 64
  val hiddenSize = 64
  val learningRate = 3e-3f
  val noiseScale = 0.15f
  val thinningProbability = 0.2f

  // -- Prepare training data --

  trait Batch derives Label

  val (initKey, thinningSeed) = Key(42).splitToTuple(2)

  val trainDataset = TwoMoons.sampleFix(numSamples, noiseScale)
  val trainDataBatchStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize)

  // -- Prepare train trajectory --

  val optimizer = Adam(learningRate = learningRate)

  /** The loss of a model, knowing nothing of any of this, at whichever parameters it is handed. */
  def costFnFor[S: Label](features: Tensor2[S, Feature, Float32], labels: Tensor1[S, Int32])(params: MoonsMLP.Params): Tensor0[Float32] =
    val model = MoonsMLP(params)
    zipvmap(Axis[S])(features, labels): (point, moon) =>
      CategoricalCrossEntropy.fromLogits(moon, model.logits(point))
    .mean

  def gradientStep(batch: TwoMoons.BatchSample[Batch], state: TrainState): TrainState =
    val (nextKey, thinningKey) = state.trainKey.split2()
    // Thin the parameters to get a perturbed model for gradient computation
    // Note we apply here the same thinning across all samples in the batch.
    val thinParams = state.params.thin(thinningProbability, thinningKey)
    val (value, grads) = Autodiff.valueAndGrad(costFnFor(batch.features, batch.labels))(thinParams)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, nextKey, value)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialState =
    val initialParams = MoonsMLP.Params.init(hiddenSize, initKey)
    val initialOptimizerState = optimizer.init(initialParams)
    TrainState(initialParams, initialOptimizerState, thinningSeed, Tensor0(-1f))

  val trainTrajectory = trainDataBatchStream.scanLeft(initialState):
    case (state, batch) =>
      jitGradientStep(batch, state)

  // -- Run train trajectory --

  val checkpointer = TensorTreeCheckpointer.newIn("out/MoonsMLP")
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)
  val finalState = trainTrajectory
    .tapEvery(100):
      // Report training loss
      case (state, step) =>
        println(trainMonitor.report(step, state))
    .tapEvery(500):
      // Save checkpoints
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Checkpoint saved at step $step")
    .drop(numIterations)
    .next()

  println(f"Final cost: ${finalState.lastCost.item}%.6f")
  println(s"Done. Wrote ${checkpointer.rootPath}.")
