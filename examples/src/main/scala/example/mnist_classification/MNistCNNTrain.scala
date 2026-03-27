package deepwit.example.mnist_classification

import dimwit.*
import dimwit.Conversions.given
import dimwit.random.Random
import deepwit.*
import examples.timed
import examples.dataset.MNISTLoader
import nn.GradientDescent
import deepwit.cnn.Conv2DLayer
import examples.dataset.MNISTBatchSample
import deepwit.logging.TenZarrLogger

private trait Batch derives Label

case class TrainState(params: MNistCNN.Params, lastCost: Tensor0[Float])

@main
def mnistCNNTrain(): Unit =

  val learningRate = 0.01f
  val numIterations = 10_000
  val batchSize = 128

  val (dataKey, trainKey) = Random.Key(42).split2()

  val trainDataset = MNISTLoader.createTrainingDataset().get
  val testDataset = MNISTLoader.createTestDataset().get

  val trainDataStream = trainDataset.toBatchStream(Axis[Batch] -> batchSize)

  val initialParams = MNistCNN.Params(trainKey)(16, 32)

  def batchLoss(batchImages: Tensor[(Batch, Height, Width), Float], batchLabels: Tensor1[Batch, Int])(
      params: MNistCNN.Params
  ): Tensor0[Float] =
    val model = MNistCNN(params)
    val batchLosses = zipvmap(Axis[Batch])(batchImages, batchLabels):
      case (img, target) =>
        CategoricalCrossEntropy.fromLogits(target, model.logits(img))
    batchLosses.mean

  def costFnFor[S: Label](images: Tensor3[S, Height, Width, Float], labels: Tensor1[S, Int])(params: MNistCNN.Params): Tensor0[Float] =
    val model = MNistCNN(params)
    zipvmap(Axis[S])(images, labels): (image, label) =>
      val logits = model.logits(image)
      CategoricalCrossEntropy.fromLogits(label, logits)
    .mean

  val optimizer = GradientDescent(learningRate = Tensor0(learningRate))

  def gradientStep(
      batch: MNISTBatchSample[Batch],
      state: TrainState
  ): TrainState =
    val grads = Autodiff.grad(costFnFor(batch.images, batch.labels))(state.params)
    val cost = costFnFor(batch.images, batch.labels)(state.params)
    val (newParams, _) = optimizer.update(grads, state.params, ())
    TrainState(newParams, cost)

  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  // Training Loop
  val trainTrajectory = trainDataStream.scanLeft(TrainState(initialParams, Tensor0(-1f))):
    case (state, batch) =>
      dimwit.gc()
      jitGradientStep(batch, state)

  // Evaluation
  def evaluate[S: Label](params: MNistCNN.Params, dataX: Tensor[(S, Height, Width), Float], dataY: Tensor1[S, Int]): Tensor0[Float] =
    val model = MNistCNN(params)
    val predictions = dataX.vmap(Axis[S])(model)
    val matches = zipvmap(Axis[S])(predictions, dataY)(_ === _)
    matches.asFloat.mean

  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val logger = new TenZarrLogger(f"out/MNistCNN/$time")
  val trainMonitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = state => state.lastCost.item)

  val state = trainTrajectory
    .tapEvery(10):
      case (state, step) =>
        println(trainMonitor.report(step, state))
    .tapEvery(500):
      case (state, step) =>
        val lossValue = costFnFor(testDataset.images, testDataset.labels)(state.params).item
        println(s"Step $step | Test loss: $lossValue")
    .tapEvery(500):
      case (state, step) =>
        logger.logTensorTree("checkpoint", step, state)
        println(s"Checkpoint saved at epoch $step")
    .drop(numIterations)
    .head
