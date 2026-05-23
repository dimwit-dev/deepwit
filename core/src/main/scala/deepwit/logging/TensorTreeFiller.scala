package deepwit.logging

import dimwit.tensor.*
import scala.deriving.*
import scala.compiletime.*

/** A typeclass for instantiating a tensor tree structure from a path-based provider.
  * Extracted from TensorTree as it deals with creation rather than manipulation.
  */
trait TensorTreeFiller[P]:
  def fill(provider: TensorTreeFiller.TensorProvider, path: String = ""): P

object TensorTreeFiller:

  /** A polymorphic function that provides a Tensor given a type, its labels, and a string path */
  type TensorProvider = [T <: Tuple, V] => (Labels[T]) ?=> String => Tensor[T, V]

  def apply[P](using ttf: TensorTreeFiller[P]): TensorTreeFiller[P] = ttf

  /** Generic instance for any Tensor[Q, V] */
  given genericTensorInstance[Q <: Tuple, V](using n: Labels[Q]): TensorTreeFiller[Tensor[Q, V]] with
    def fill(provider: TensorProvider, path: String = ""): Tensor[Q, V] =
      provider[Q, V](using n)(path)

  /** Instance for Unit (useful for optimizers with no internal state) */
  given TensorTreeFiller[Unit] with
    def fill(provider: TensorProvider, path: String = ""): Unit = ()

  /** Instance for a tuple of two tensors */
  given tupleInstance[P1, P2](using t1: TensorTreeFiller[P1], t2: TensorTreeFiller[P2]): TensorTreeFiller[(P1, P2)] with
    def fill(provider: TensorProvider, path: String = ""): (P1, P2) =
      val p1Path = if path.isEmpty then "_1" else s"$path._1"
      val p2Path = if path.isEmpty then "_2" else s"$path._2"
      (t1.fill(provider, p1Path), t2.fill(provider, p2Path))

  /** Automatically derive a TensorTreeFiller instance for any case class (or product type) */
  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): TensorTreeFiller[P] =
    val elemInstances = summonAll[Tuple.Map[m.MirroredElemTypes, TensorTreeFiller]]
    val instances = elemInstances.toList.asInstanceOf[List[TensorTreeFiller[Any]]]
    val fieldNames = constValueTuple[m.MirroredElemLabels].toList.map(_.toString)
    derivedImpl(instances, fieldNames, m)

  private def derivedImpl[P <: Product](
      instances: List[TensorTreeFiller[Any]],
      fieldNames: List[String],
      m: Mirror.ProductOf[P]
  ): TensorTreeFiller[P] = new TensorTreeFiller[P]:
    def fill(provider: TensorProvider, path: String = ""): P =
      val generatedElems = instances.zip(fieldNames).map:
        case (inst, fieldName) =>
          val newPath = if path.isEmpty then fieldName else s"$path.$fieldName"
          inst.fill(provider, newPath)
      m.fromProduct(Tuple.fromArray(generatedElems.map(_.asInstanceOf[Object]).toArray))
