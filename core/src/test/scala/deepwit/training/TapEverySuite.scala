package deepwit.training

import scala.collection.mutable.ListBuffer
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TapEverySuite extends AnyFunSpec with Matchers:

  describe("Iterator.tapEvery"):

    it("fires at every n-th index but not at zero"):
      val seen = ListBuffer.empty[(String, Int)]
      Iterator.from(0).map(i => s"e$i").tapEvery(3)((t, id) => seen += ((t, id))).take(10).toList
      seen.toList shouldBe List(("e3", 3), ("e6", 6), ("e9", 9))

    it("passes the elements through unchanged"):
      val result = Iterator.from(0).tapEvery(2)((_, _) => ()).take(5).toList
      result shouldBe List(0, 1, 2, 3, 4)

    it("does not fire beyond what is consumed"):
      val seen = ListBuffer.empty[Int]
      Iterator.from(0).tapEvery(1)((_, id) => seen += id).take(3).toList
      seen.toList shouldBe List(1, 2)

  describe("LazyList.tapEvery"):

    it("fires at every n-th index but not at zero"):
      val seen = ListBuffer.empty[(String, Int)]
      LazyList.from(0).map(i => s"e$i").tapEvery(3)((t, id) => seen += ((t, id))).take(10).toList
      seen.toList shouldBe List(("e3", 3), ("e6", 6), ("e9", 9))

    it("stays lazy until the elements are forced"):
      val seen = ListBuffer.empty[Int]
      val tapped = LazyList.from(0).tapEvery(1)((_, id) => seen += id)
      seen.toList shouldBe empty
      tapped.take(3).toList shouldBe List(0, 1, 2)
