package deepwit.example

import dimwit.*
import dimwit.Conversions.given
import deepwit.*
import dimwit.stats.Normal
import dimwit.random.Random
import nn.Adam
import deepwit.logging.TenZarrLogger

trait Batch derives Label
trait Input derives Label
trait Hidden derives Label
trait Output derives Label

case class MLPParams(
    hiddenLayer: AffineLayer.Params[Input, Hidden],
    outputLayer: AffineLayer.Params[Hidden, Output]
)

class MLP(params: MLPParams):

  private val hiddenLayer = AffineLayer(params.hiddenLayer)
  private val outputLayer = AffineLayer(params.outputLayer)

  def apply(input: Tensor1[Input, Float]): Tensor1[Output, Float] =
    val hidden = relu(hiddenLayer(input))
    outputLayer(hidden)

val batchExtent = Axis[Batch] -> 32
val inputExtent = Axis[Input] -> 10
val hiddenExtent = Axis[Hidden] -> 20
val outputExtent = Axis[Output] -> 1

@main def trainMLPWithLogging(): Unit =
  // 1. Setup Logging
  val time = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val logger = new TenZarrLogger(f"out/$time")

  // 2. Define Dimensions
  val inputExtent = Axis[Input] -> 10
  val hiddenExtent = Axis[Hidden] -> 20
  val outputExtent = Axis[Output] -> 1
  val batchExtent = Axis[Batch] -> 32

  // 3. Initialize Parameters
  val key = Random.Key.fromTime()
  val (paramsKey, dataKey, trainKey) = key.splitToTuple(3)
  val (hiddenKey, outputKey) = paramsKey.splitToTuple(2)

  var mlpParams = MLPParams(
    AffineLayer.Params.xavierNormal(inputExtent, hiddenExtent, hiddenKey),
    AffineLayer.Params.xavierNormal(hiddenExtent, outputExtent, outputKey)
  )

  // 4. Synthetic Data Generation (y = 3x + noise)
  val trueWeight = Tensor(Shape(inputExtent, outputExtent)).fill(3.0f)
  def generateBatch(k: Random.Key): (Tensor2[Batch, Input, Float], Tensor2[Batch, Output, Float]) =
    val (k1, k2) = k.splitToTuple(2)
    val x = Normal(Tensor(Shape(batchExtent, inputExtent)).fill(0f), Tensor(Shape(batchExtent, inputExtent)).fill(1f)).sample(k1)
    val noise = Normal(Tensor(Shape(batchExtent, outputExtent)).fill(0f), Tensor(Shape(batchExtent, outputExtent)).fill(0.1f)).sample(k2)
    val y = (x.dot(Axis[Input])(trueWeight)) + noise
    (x, y)

  // 5. Training Setup
  val optimizer = Adam(learningRate = 1e-3f)
  var optState = optimizer.init(mlpParams)

  def lossFn(p: MLPParams, x: Tensor2[Batch, Input, Float], y: Tensor2[Batch, Output, Float]): Tensor0[Float] =
    val model = MLP(p)
    val pred = x.vmap(Axis[Batch])(model.apply)
    (pred - y).pow(2f).mean

  println("Starting Training...")

  val dataKeys = dataKey.split(1000)
  for i <- 0 until 1000 do
    val (batchX, batchY) = generateBatch(dataKeys(i))

    // Compute Gradients and Update
    val currentLoss = lossFn(mlpParams, batchX, batchY)
    val grads = Autodiff.grad(p => lossFn(p, batchX, batchY))(mlpParams)
    logger.logTensorTree(s"train/grad", i, grads)

    val (nextParams, nextOptState) = optimizer.update(grads, mlpParams, optState)
    logger.logTensorTree(s"train/params", i, nextParams)

    mlpParams = nextParams
    optState = nextOptState

    // 6. LOGGING TO ZARR
    // We log the raw weights and the current loss
    if i % 10 == 0 then
      println(f"Iteration $i - Loss: ${currentLoss.item}%.4f")

      // Log Scalar Loss
      logger.logTensor("train/loss", i, currentLoss)

  println("Training complete. Data saved to mlp_experiment.zarr")
