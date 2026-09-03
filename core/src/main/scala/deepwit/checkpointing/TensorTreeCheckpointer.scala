package deepwit.checkpointing

import java.nio.file.{Files, Path}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import scala.util.Try

import dimwit.tensortree.TensorTree
import dimwit.tensortree.{TensorTreeIO, TensorTreeFormat}

/** Holds the checkpoints of one training run: a folder of trees, one file per iteration. */
class TensorTreeCheckpointer(
    val rootPath: String,
    private val format: TensorTreeFormat = TensorTreeFormat.Pickle,
    private val overwrite: Boolean = false
):

  assert(rootPath.nonEmpty, "Root path must be non-empty.")

  private var initialized: Boolean = false

  def save[Tree: TensorTree](tree: Tree, iteration: Int): Unit =
    require(iteration >= 0, s"Iteration must be non-negative, but got $iteration")
    if !initialized then
      createFolder(Path.of(rootPath))
      initialized = true
    val path = Path.of(s"$rootPath/$iteration.pkl")
    TensorTreeIO.save(tree, path)

  /** @param iteration The iteration whose checkpoint to read.
    * @return The tree saved at `iteration`. `None` means absence. Wrong TensorTree structure throws exception.
    */
  def load[Tree: TensorTree](iteration: Int): Option[Tree] =
    require(iteration >= 0, s"Iteration must be non-negative, but got $iteration")
    val path = Path.of(s"$rootPath/$iteration.pkl")
    Option.when(Files.isRegularFile(path))(TensorTreeIO.load[Tree](path))

  /** The tree at the furthest iteration saved here, or `None` when there is none. */
  def loadLatest[Tree: TensorTree]: Option[Tree] =
    iterations.lastOption.flatMap(load[Tree])

  def iterations: Seq[Int] =
    val root = Path.of(rootPath)
    if !root.toFile.exists() then
      Seq.empty
    else
      root.toFile.listFiles()
        .filter(_.isFile)
        .map(_.getName)
        .filter(_.endsWith(".pkl"))
        .map: name =>
          name.stripSuffix(".pkl").toInt
        .sorted
        .toIndexedSeq

  private def createFolder(path: Path): Unit =
    assert(!path.toFile.exists() || overwrite, s"Directory $rootPath already exists. Set overwrite = true to overwrite.")
    path.toFile.mkdirs()

object TensorTreeCheckpointer:

  /** How a folder is named, so that ordering folders by name orders them by time. */
  private val runFolder = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")

  /** A checkpointer for a fresh folder under `root`, named for the moment it was made. */
  def newIn(root: String, format: TensorTreeFormat = TensorTreeFormat.Pickle): TensorTreeCheckpointer =
    TensorTreeCheckpointer(s"$root/${LocalDateTime.now().format(runFolder)}", format)

  /** The newest folder under `root` that [[newIn]] could have made, or `None` if there is none. */
  def latestIn(root: String, format: TensorTreeFormat = TensorTreeFormat.Pickle): Option[TensorTreeCheckpointer] =
    def isRunFolder(name: String): Boolean =
      Try(LocalDateTime.parse(name, runFolder)).isSuccess

    Option(Path.of(root).toFile.listFiles())
      .getOrElse(Array.empty[java.io.File])
      .filter(file => file.isDirectory && isRunFolder(file.getName))
      .map(_.getPath)
      .maxOption
      .map(TensorTreeCheckpointer(_, format))
