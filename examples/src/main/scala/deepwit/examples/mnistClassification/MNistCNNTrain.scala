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

private trait Batch derives Label

case class TrainState(
    params: MNistCNN.Params,
    optimizerState: GradientDescentState[MNistCNN.Params],
    trainKey: Key,
    lastCost: Tensor0[Float32]
)

@main
def mnistCNNTrain(): Unit =

  val numIterations = 10_000
  val batchSize = 128
  val learningRate = 0.01f
  val dropoutProbability = 0.2f

  val (paramsKey, dropoutSeed) = Random.Key(42).split2()

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val testDataset = MNISTLoader.createTestDataset().get

  val trainDataBatchStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize)

  def costFnFor[S: Label](images: Tensor3[S, Height, Width, Float32], labels: Tensor1[S, Int32])(params: MNistCNN.Params): Tensor0[Float32] =
    val model = MNistCNN(params)
    zipvmap(Axis[S])(images, labels): (image, label) =>
      val logits = model.logits(image)
      CategoricalCrossEntropy.fromLogits(label, logits)
    .mean

  val optimizer = GradientDescent(learningRate = learningRate)

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

  val initialParams = MNistCNN.Params(paramsKey)(16, 32)
  val initialOptimizerState = optimizer.init(initialParams)

  val trainTrajectory = trainDataBatchStream.scanLeft(TrainState(initialParams, initialOptimizerState, Random.Key(42), Tensor0(-1f))):
    case (state, batch) =>
      dimwit.gc()
      jitGradientStep(batch, state)

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val checkpointer = new TensorTreeCheckpointer(f"out/MNistCNN/$time")
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)

  val state = trainTrajectory
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
