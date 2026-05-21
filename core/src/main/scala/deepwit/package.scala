import scala.languageFeature.experimental.macros
package object deepwit:
  export deepwit.base.{AffineLayer, LinearLayer}
  export deepwit.base.ActivationFunction.{gelu, relu, sigmoid, softmax}
  export deepwit.cnn.{Conv2DLayer, AffineConv2DLayer, LinearConv2DLayer}
  export deepwit.embedder.{ConvImageToPatchEmbedder, LearnedAbsolutePositionalInjector, VocabularyEmbedder}
  export deepwit.transformer.{MLPEmbeddingMixer, Transformer, TransformerLayer, CrossTransformer, CrossTransformerLayer, CausalTransformer, BidirectionalTransformer}
  export deepwit.transformer.{causalMask, identityMask}
  export deepwit.transformer.attention.{SelfAttention, CrossAttention, MultiHeadSelfAttention, MultiHeadCrossAttention}
  export deepwit.init.{xavierNormal, xavierUniform}
  export deepwit.normalization.{LayerNorm, RMSNorm}

  // Dropout thinning
  export deepwit.regularization.{sampleThinAffineLayer, sampleThinLearnedAbsolutePositionalInjector, sampleThinLinearLayer, sampleThinProjection, sampleThinVocabularyEmbedder}

  export deepwit.loss.{BinaryCrossEntropy, CategoricalCrossEntropy, BernoulliCrossEntropy}

  object labels:
    export deepwit.transformer.MLPEmbeddingMixer.EmbeddingMixed
    export deepwit.transformer.attention.{Head, HeadKey, HeadQuery, HeadValue}
    export deepwit.transformer.attention.{Query, Key, Value, AttentionWeights}

  // Train utils

  trait Monitor[S]:
    def report(step: Int, state: S): String

  object Monitor:

    def default[S](batchSize: Int, lossLens: S => Float): Monitor[S] =
      ConcatMonitor(List(
        StepMonitor(),
        LossMonitor(lossLens),
        PerformanceMonitor(batchSize)
      ))

    case class ConcatMonitor[S](monitors: List[Monitor[S]], sep: String = " | ") extends Monitor[S]:
      def report(step: Int, state: S): String =
        monitors.map(m => m.report(step, state)).mkString(sep)

    case class StepMonitor[S]() extends Monitor[S]:
      def report(step: Int, state: S): String =
        f"Step $step"

    case class LossMonitor[S](lossLens: S => Float) extends Monitor[S]:
      def report(step: Int, state: S): String =
        f"Loss: ${lossLens(state)}%.4f"

    case class PerformanceMonitor[S](batchSize: Int) extends Monitor[S]:
      private var lastTime = System.nanoTime()
      def report(step: Int, state: S): String =
        val currentTime = System.nanoTime()
        val elapsedS = (currentTime - lastTime) / 1e9d
        val sPerSec = batchSize / elapsedS
        lastTime = currentTime
        f"$sPerSec%.2f samples/sec"

  extension [T](it: Iterator[T])

    def tapEvery(n: Int)(f: (T, Int) => Unit): Iterator[T] =
      it
        .zipWithIndex
        .tapEach: (t, id) =>
          if id > 0 && id % n == 0 then f(t, id)
        .map(_._1)

  extension [T](it: LazyList[T])

    def tapEvery(n: Int)(f: (T, Int) => Unit): LazyList[T] =
      it
        .zipWithIndex
        .tapEach: (t, id) =>
          if id > 0 && id % n == 0 then f(t, id)
        .map(_._1)

  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.FloatTree
  import dimwit.FloatTree.{map, mapLeaves}

  extension [H, V: IsFloating](grads: Grad[H])(using FloatTree[H, V], TensorTree[H])
    def clipGlobalNorm(maxNorm: Tensor0[V], epsilon: Double = 1e-6): Grad[H] =
      val gradNorm = grads.value.mapLeaves([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x.pow(2).sum).reduce(_ + _).sqrt
      val scale = minimum(1f, maxNorm / (gradNorm + epsilon))
      Grad(grads.value.map([T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x *! scale))
