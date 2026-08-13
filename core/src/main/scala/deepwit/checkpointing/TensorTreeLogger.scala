package deepwit.checkpointing

import java.nio.file.Path
import dimwit.tensortree.TensorTree
import dimwit.tensortree.{TensorTreeIO, TensorTreeFormat}
import java.io.FileNotFoundException

class TensorTreeCheckpointer(
    private val rootPath: String,
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

  def load[Tree: TensorTree](iteration: Int): Option[Tree] =
    require(iteration >= 0, s"Iteration must be non-negative, but got $iteration")
    Some: // TODO TensorTreeIO.load must return Option
      val path = Path.of(s"$rootPath/$iteration.pkl")
      TensorTreeIO.load[Tree](path)

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

  private def createFolder(path: Path): Unit =
    assert(!path.toFile.exists() || overwrite, s"Directory $rootPath already exists. Set overwrite = true to overwrite.")
    path.toFile.mkdirs()
