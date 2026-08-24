package deepwit.examples.mnistClassification

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.GradientDescent

import deepwit.loss.CategoricalCrossEntropy

import deepwit.examples.dataset.{MNISTLoader, MNISTBatchSample}
import dimwit.optimizer.GradientDescentState
import deepwit.training.{Monitor, tapEvery}
import deepwit.checkpointing.TensorTreeCheckpointer

case class TrainState(
    params: MNistCNN.Params,
    optimizerState: GradientDescentState[MNistCNN.Params],
    lastCost: Tensor0[Float32]
)

@main
def train(): Unit =

  // -- Configuration --

  val numIterations = 10_000
  val batchSize = 128
  val learningRate = 0.01f

  // -- Prepare train trajectory --

  trait Batch derives Label

  val initKey = Key(42)

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
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch.images, batch.labels))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialState =
    val initialParams = MNistCNN.Params.init(16, 32, initKey)
    val initialOptimizerState = optimizer.init(initialParams)
    TrainState(initialParams, initialOptimizerState, Tensor0(-1f))

  val trainTrajectory = trainDataBatchStream.scanLeft(initialState):
    case (state, batch) =>
      jitGradientStep(batch, state)

  // -- Run train trajectory --

  val checkpointer = TensorTreeCheckpointer.newIn("out/MNistCNN")
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)
  val finalState = trainTrajectory
    .tapEvery(10):
      // Print training progress
      case (state, step) => println(trainMonitor.report(step, state))
    .tapEvery(500):
      // Evaluate on test dataset
      case (state, step) =>
        val lossValue = costFnFor(testDataset.images, testDataset.labels)(state.params).item
        println(s"Step $step | Test loss: $lossValue")
    .tapEvery(500):
      // Save checkpoint
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .next()

  println(s"Done. Wrote ${checkpointer.rootPath}.")
