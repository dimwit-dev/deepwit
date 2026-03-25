package examples.dataset

import dimwit.*
import dimwit.Conversions.given

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import java.io.RandomAccessFile
import scala.util.Try

object MNISTLoader:

  trait Sample derives Label
  trait TrainSample extends Sample derives Label
  trait TestSample extends Sample derives Label
  trait Height derives Label
  trait Width derives Label

  private val pythonLoader = py.eval("lambda b64, shape: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=__import__('numpy').uint8).reshape(shape).astype(__import__('numpy').int32))")

  def loadImages[S <: Sample: Label](filename: String): Tensor3[S, Height, Width, Int] =
    val file = new RandomAccessFile(filename, "r")
    try
      val magic = file.readInt()
      if magic != 2051 then throw new IllegalArgumentException(s"Invalid magic: $magic")

      val totalImages = file.readInt()
      val rows = file.readInt()
      val cols = file.readInt()
      assert(rows == 28 && cols == 28, s"Invalid image dimensions: $rows x $cols")

      val totalPixels = totalImages * rows * cols

      val pixels = new Array[Byte](totalPixels)
      file.readFully(pixels)

      val shape = Shape(Axis[S] -> totalImages, Axis[Height] -> rows, Axis[Width] -> cols)

      // MNIST pixels are unsigned bytes
      // So we read them as Byte and interpret as UInt8 when creating the Tensor
      given ExecutionType[Byte] = ExecutionTypeFor[Byte](DType.UInt8)
      Tensor(shape).fromArray(pixels)

    finally
      file.close()

  def loadLabels[S <: Sample: Label](filename: String): Tensor1[S, Int] =
    val file = new RandomAccessFile(filename, "r")
    try
      val magic = file.readInt()
      if magic != 2049 then throw new IllegalArgumentException(s"Invalid magic for labels: $magic (expected 2049)")

      val totalLabels = file.readInt()

      val labels = new Array[Byte](totalLabels)
      file.readFully(labels)

      val shape = Shape(Axis[S] -> totalLabels)
      Tensor(shape).fromArray(labels)

    finally
      file.close()

  private def createDataset[S <: Sample: Label](imagesFile: String, labelsFile: String): Try[Tuple2[Tensor[(S, Height, Width), Float], Tensor1[S, Int]]] =
    Try:
      val images = loadImages[S](imagesFile)
      val labels = loadLabels[S](labelsFile)
      require(images.shape(Axis[S]) == labels.shape(Axis[S]), s"Number of images and labels must match")
      val imagesFloat = images.asFloat /! 255.0f
      (imagesFloat, labels)

  def createTrainingDataset(dataDir: String = "data"): Try[Tuple2[Tensor[(TrainSample, Height, Width), Float], Tensor1[TrainSample, Int]]] =
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    createDataset[TrainSample](imagesFile, labelsFile)

  def createTestDataset(dataDir: String = "data"): Try[Tuple2[Tensor[(TestSample, Height, Width), Float], Tensor1[TestSample, Int]]] =
    val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
    val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
    createDataset[TestSample](imagesFile, labelsFile)
