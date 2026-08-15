package deepwit.examples.gpt

import java.io.File
import dimwit.*
import dimwit.Conversions.given
import dimwit.python.PyBridge.liftPyTensor
import me.shadaj.scalapy.py
import dimwit.stats.Uniform
import dimwit.hardware.DeviceBackend.GPU

object FineWebDataset:

  case class BatchSample(
      targets: Tensor2[Sample, Context, Int32],
      inputs: Tensor2[Sample, Context, Int32]
  )

  opaque type LazyTensor1[L, V] = Tensor1[L, V]

  lazy val np = py.module("numpy")

  // 1. Updated to skip the 1024-byte header
  def loadShard(binaryPath: String): LazyTensor1[Sample, UInt16] =
    liftPyTensor(
      np.memmap(binaryPath, dtype = np.uint16, mode = "r", offset = 1024)
    )

  def loadBatch(data: LazyTensor1[Sample, UInt16], batchSize: Int, contextLength: Int, key: Random.Key): BatchSample =
    val maxIdx = data.shape(Axis[Sample]) - batchSize - 1
    val randomIndices = IndependentDistribution.fromUnivariate(
      Shape1(Axis[Sample] -> batchSize),
      Uniform(Tensor0(0), Tensor0(maxIdx))
    ).sample(key)
    val shiftedIndices = randomIndices +! 1
    val inputs = randomIndices.vmap(Axis[Sample])(startIndex =>
      data.dynamicSlice(startIndex, contextLength).relabelTo(Axis[Context])
    )
    val targets = shiftedIndices.vmap(Axis[Sample])(startIndex =>
      data.dynamicSlice(startIndex, contextLength).relabelTo(Axis[Context])
    )
    val gpu = GPU.devices.head
    BatchSample(targets.asInt32.toDevice(gpu), inputs.asInt32.toDevice(gpu))

  def batchStream(dataDir: String, filePrefix: String, batchSize: Int, contextLength: Int, initialKey: Random.Key): Iterator[BatchSample] =
    val files = new File(dataDir)
      .listFiles()
      .filter(_.getName.startsWith(filePrefix))
      .sortBy(_.getName)

    if files.isEmpty then throw new Exception(s"No files found matching prefix '$filePrefix' in $dataDir")

    // This Iterator will cycle through files indefinitely
    Iterator.continually(files.indices).flatten.flatMap { fileIdx =>
      val file = files(fileIdx % files.length)
      val data = loadShard(file.getAbsolutePath)
      val shardTokens = data.shape(Axis[Sample])
      val batchesInShard = shardTokens / (batchSize * contextLength)

      // Using an iterator here to generate batches without caching
      // TODO The key stream restarts from initialKey for every shard, so every shard replays the
      //      identical sequence of sample offsets. Thread one key stream through all shards instead.
      Iterator.iterate(initialKey)(_.split2()._2) // Infinite stream of keys
        .map(key => loadBatch(data, batchSize, contextLength, key))
        .take(batchesInShard)
    }
