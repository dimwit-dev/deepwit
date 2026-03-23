package example

import dimwit.Label

package object gpt:

  trait Sample derives Label
  trait Vocab derives Label
  trait Embedding derives Label
  trait Context derives Label

  case class Timer private (
      private val decay: Float = 0.01f
  ):

    private var lastTime = System.currentTimeMillis()
    private var internalRunningAverage = -1f

    def tick(): Unit =
      val now = System.currentTimeMillis()
      val elapsed = now - lastTime
      internalRunningAverage =
        if internalRunningAverage == -1f
        then elapsed
        else internalRunningAverage * decay + elapsed * (1f - decay)
      lastTime = now

    def reset(): Unit =
      lastTime = System.currentTimeMillis()
      internalRunningAverage = -1f

    def runningAvgSeconds: Float = internalRunningAverage / 1000f

  object Timer:
    def start(): Timer = new Timer()

  extension [T](it: LazyList[T])
    def tapEvery(n: Int)(f: (T, Int) => Unit): LazyList[T] =
      it
        .zipWithIndex
        .tapEach: (t, id) =>
          if id % n == (n - 1) then f(t, id)
        .map(_._1)
