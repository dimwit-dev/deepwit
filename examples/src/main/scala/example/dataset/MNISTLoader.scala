package examples.dataset

import dimwit.*
import dimwit.Conversions.given

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import java.io.RandomAccessFile
import scala.util.Try

import MNISTLoader.{Width, Height}

case class MNISTBatchSample[Batch](images: Tensor[(Batch, Height, Width), Float32], labels: Tensor1[Batch, Int32])

case class MNISTDataset[S: Label](images: Tensor3[S, Height, Width, Float32], labels: Tensor1[S, Int32]):

  def toBatchStream[Batch: Label](
      batchExtent: AxisExtent[Batch]
  ): Iterator[MNISTBatchSample[Batch]] =
    val totalSamples = images.shape(Axis[S])
    val batchSize = batchExtent.size
    Iterator.iterate(0)(_ + batchSize).map: offset =>
      val batchIds = (0 until batchSize).map(i => (offset + i) % totalSamples)
      val batchImages = images.slice(Axis[S].at(batchIds)).relabel(Axis[S], Axis[Batch])
      val batchLabels = labels.slice(Axis[S].at(batchIds)).relabel(Axis[S], Axis[Batch])
      MNISTBatchSample(batchImages, batchLabels)

object MNISTLoader:

  trait Sample derives Label
  trait TrainSample extends Sample derives Label
  trait TestSample extends Sample derives Label
  trait Height derives Label
  trait Width derives Label

  private val pythonLoader = py.eval("lambda b64, shape: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=__import__('numpy').uint8).reshape(shape).astype(__import__('numpy').int32))")

  def loadImages[S <: Sample: Label](filename: String): Tensor3[S, Height, Width, UInt8] =
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
      Tensor(shape, VType[UInt8]).fromArray(pixels)

    finally
      file.close()

  def loadLabels[S <: Sample: Label](filename: String): Tensor1[S, UInt8] =
    val file = new RandomAccessFile(filename, "r")
    try
      val magic = file.readInt()
      if magic != 2049 then throw new IllegalArgumentException(s"Invalid magic for labels: $magic (expected 2049)")

      val totalLabels = file.readInt()

      val labels = new Array[Byte](totalLabels)
      file.readFully(labels)

      val shape = Shape(Axis[S] -> totalLabels)
      Tensor(shape, VType[UInt8]).fromArray(labels)

    finally
      file.close()

  private def createDataset[S <: Sample: Label](imagesFile: String, labelsFile: String): Try[MNISTDataset[S]] =
    Try:
      val images = loadImages[S](imagesFile)
      val labels = loadLabels[S](labelsFile)
      require(images.shape(Axis[S]) == labels.shape(Axis[S]), s"Number of images and labels must match")
      val imagesFloat = images.asFloat32 /! 255.0f
      MNISTDataset(imagesFloat, labels.asInt32)

  def createTrainingDataset(dataDir: String = "data"): Try[MNISTDataset[TrainSample]] =
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    createDataset[TrainSample](imagesFile, labelsFile)

  def createTestDataset(dataDir: String = "/data"): Try[MNISTDataset[TestSample]] =
    val path = getClass.getResource(dataDir).getPath()
    val imagesFile = s"$path/t10k-images-idx3-ubyte"
    val labelsFile = s"$path/t10k-labels-idx1-ubyte"
    createDataset[TestSample](imagesFile, labelsFile)
