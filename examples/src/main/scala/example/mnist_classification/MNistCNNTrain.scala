package deepwit.example.mnist_classification

import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.GradientDescent

import deepwit.*

import examples.timed
import examples.dataset.{MNISTLoader, MNISTBatchSample}
import dimwit.optimizer.GradientDescentState

private trait Batch derives Label

case class TrainState(params: MNistCNN.Params, optimizerState: GradientDescentState[MNistCNN.Params], lastCost: Tensor0[Float32])

@main
def mnistCNNTrain(): Unit =

  val numIterations = 10_000
  val batchSize = 128
  val learningRate = 0.01f

  val trainKey = Random.Key(42)

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
    val (cost, grads) = Autodiff.valueAndGrad(costFnFor(batch.images, batch.labels))(state.params)
    val (newParams, newOptimizerState) = optimizer.update(grads, state.params, state.optimizerState)
    TrainState(newParams, newOptimizerState, cost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val initialParams = MNistCNN.Params(trainKey)(16, 32)
  val initialOptimizerState = optimizer.init(initialParams)
  val trainTrajectory = trainDataBatchStream.scanLeft(TrainState(initialParams, initialOptimizerState, Tensor0(-1f))):
    case (state, batch) =>
      dimwit.gc()
      jitGradientStep(batch, state)

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val logger = new TensorTreeLogger(f"out/MNistCNN/$time")
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)

  val state = trainTrajectory
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
        logger.save(state, step)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .next()
