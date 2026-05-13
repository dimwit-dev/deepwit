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

  def logTensorTree[Data: TensorTree](name: String, iteration: Int, data: Data): Unit =
    ???
    /*TensorTree.foreach(
      data,
      [T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (tensorName: String, tensor: Tensor[T, V]) =>
            logTensor(s"$name/$iteration/$tensorName", tensor)
    )*/

  def logTensor[T <: Tuple, V](name: String, data: Tensor[T, V]): Unit =
    assert(!root.__contains__(name).as[Boolean])
    val pyData = np.array(toPyTensor(data))

    val dset = root.create_dataset(
      name = name,
      shape = data.shape.dimensions.toPythonCopy,
      dtype = "f4"
    )

    if data.shape.dimensions.isEmpty
    then dset.bracketUpdate(py.Dynamic.global.tuple(), pyData)
    else dset.bracketUpdate(py.Dynamic.global.slice(py.None), pyData)

  def loadTensorTree[Data: TensorTree](name: String, iteration: Int): Option[Data] =
    val iterationPath = s"$name/$iteration"
    Option.when(root.__contains__(iterationPath).as[Boolean]):
      ???
      /*
      TensorTree[Data].fill([T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (tensorName: String) =>
            val zarrArray = root.bracketAccess(s"$iterationPath/$tensorName")
            val shape = zarrArray.shape.as[Seq[Int]]
            val pyTensor =
              if shape.isEmpty then zarrArray.bracketAccess(me.shadaj.scalapy.py.Dynamic.global.Ellipsis)
              else zarrArray.bracketAccess(py.Dynamic.global.slice(py.None))
            liftPyTensor[T, V](pyTensor)
      )*/

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
