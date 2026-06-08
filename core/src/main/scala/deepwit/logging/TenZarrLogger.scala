package deepwit.logging

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.*
import dimwit.python.PyBridge.{toPyTensor, liftPyTensor, liftPyTensor1}

trait Iteration derives Label

class TenZarrLogger(storePath: String):

  private val zarr = py.module("zarr")
  private val np = py.module("numpy")
  private val root = zarr.open(storePath, mode = "a")

  def iterations(name: String = "checkpoint"): List[Int] =
    if !root.__contains__(name).as[Boolean] then return Nil
    val group = root.bracketAccess(name)
    val keys = py.Dynamic.global.list(group.keys()).as[Seq[String]]
    keys.flatMap(_.toIntOption).sorted.toList

  def logMetadata(name: String, data: Any): Unit =
    // TODO
    ???

  def logTensorTree[Data: TensorTree](name: String, data: Data): Unit =
    TensorTree[Data].foreachWithName(
      data,
      [T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (tensorName: String, tensor: Tensor[T, V]) =>
            logTensor(s"$name/$tensorName", tensor, overwrite = true)
    )

  def logTensorTree[Data: TensorTree](name: String, iteration: Int, data: Data): Unit =
    TensorTree[Data].foreachWithName(
      data,
      [T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (tensorName: String, tensor: Tensor[T, V]) =>
            logTensor(s"$name/$iteration/$tensorName", tensor, overwrite = false)
    )

  def logTensor[T <: Tuple, V](name: String, data: Tensor[T, V], overwrite: Boolean = false): Unit =
    if !overwrite then assert(!root.__contains__(name).as[Boolean])
    val pyData = np.array(toPyTensor(data)) // TODO I think this leads to a OOM on "classic" RAM

    val dset = root.create_dataset(
      name = name,
      shape = data.shape.dimensions.toPythonCopy,
      dtype = "f4",
      overwrite = overwrite
    )

    if data.shape.dimensions.isEmpty
    then dset.bracketUpdate(py.Dynamic.global.tuple(), pyData)
    else dset.bracketUpdate(py.Dynamic.global.slice(py.None), pyData)

  def loadTensorTree[Data: TensorTree](tree: Data, name: String, iteration: Int): Option[Data] =
    loadTensorTreeFromPath(tree, s"$name/$iteration")

  def loadTensorTree[Data: TensorTree](tree: Data, name: String): Option[Data] =
    loadTensorTreeFromPath(tree, s"$name")

  def loadTensorTreeFromPath[Data: TensorTree](tree: Data, path: String): Option[Data] =
    Option.when(root.__contains__(path).as[Boolean]):
      summon[TensorTree[Data]].mapWithName(
        tree,
        [T <: Tuple, V] =>
          (labels: Labels[T]) ?=>
            (tensorName: String, dummy: Tensor[T, V]) =>
              val zarrArray = root.bracketAccess(s"$path/$tensorName")
              val shape = zarrArray.shape.as[Seq[Int]]
              val pyTensor =
                if shape.isEmpty then zarrArray.bracketAccess(me.shadaj.scalapy.py.Dynamic.global.Ellipsis)
                else zarrArray.bracketAccess(py.Dynamic.global.slice(py.None))
              liftPyTensor[T, V](pyTensor)
      )

  def loadTree[Data: TensorTree](name: String, iteration: Int): Option[Data] =
    Option.when(root.as[py.Dynamic].__contains__(name).as[Boolean]):
      val pyData = root.bracketAccess(name)
      ???
      /*TensorTree[Data].fill([T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (path: String) =>
            val raw = pyData.bracketAccess(path).bracketAccess("values").bracketAccess(iteration)
            liftPyTensor[T, V](raw)
      )*/
