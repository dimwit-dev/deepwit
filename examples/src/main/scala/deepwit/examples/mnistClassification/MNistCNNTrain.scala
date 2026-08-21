package deepwit.examples.mnistClassification

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.GradientDescent

import deepwit.loss.CategoricalCrossEntropy

import deepwit.examples.dataset.{MNISTLoader, MNISTBatchSample}
import dimwit.optimizer.GradientDescentState
import deepwit.training.{Monitor, tapEvery}
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.regularization.Dropout

case class TrainState(
    params: MNistCNN.Params,
    optimizerState: GradientDescentState[MNistCNN.Params],
    trainKey: Key,
    lastCost: Tensor0[Float32]
)

@main
def train(): Unit =

  // -- Configuration --

  val numIterations = 10_000
  val batchSize = 128
  val learningRate = 0.01f
  val dropoutProbability = 0.2f

  // -- Prepare train trajectory --

  trait Batch derives Label

  val (initKey, dropoutSeed) = Key(42).split2()

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val testDataset = MNISTLoader.createTestDataset().get

  val trainDataBatchStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize)
  val optimizer = GradientDescent(learningRate = learningRate)

  def costFnFor[S: Label](images: Tensor3[S, Height, Width, Float32], labels: Tensor1[S, Int32])(params: MNistCNN.Params): Tensor0[Float32] =
    val model = MNistCNN(params)
    zipvmap(Axis[S])(images, labels): (image, label) =>
      val logits = model.logits(image)
      CategoricalCrossEntropy.fromLogits(label, logits)
    .mean

  def gradientStep(
      batch: MNISTBatchSample[Batch],
      state: TrainState
  ): TrainState =
    val (nextKey, dropoutKey) = state.trainKey.split2()
    val thinnedParams = state.params.thinned(dropoutProbability, dropoutKey)
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch.images, batch.labels))(thinnedParams) // use thinned parameters for the gradient computation
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState) // update the full parameters
    TrainState(newParams, newOptimizerState, nextKey, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialState =
    val initialParams = MNistCNN.Params.init(16, 32, initKey)
    val initialOptimizerState = optimizer.init(initialParams)
    TrainState(initialParams, initialOptimizerState, dropoutSeed, Tensor0(-1f))

  val trainTrajectory = trainDataBatchStream.scanLeft(initialState):
    case (state, batch) =>
      jitGradientStep(batch, state)

  // -- Run train trajectory --

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val checkpointPath = f"out/MNistCNN/$time"
  val checkpointer = new TensorTreeCheckpointer(checkpointPath)
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)
  val finalState = trainTrajectory
    .tapEvery(10):
      // Print training progress
      case (state, step) => println(trainMonitor.report(step, state))
    .tapEvery(500):
      // Evaluate on test dataset
      case (state, step) =>
        // Measured through the stored parameters, whose projection is still the identity.
        val lossValue = costFnFor(testDataset.images, testDataset.labels)(state.params).item
        println(s"Step $step | Test loss: $lossValue")
    .tapEvery(500):
      // Save checkpoint
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .next()

  println(s"Done. Wrote $checkpointPath.")
