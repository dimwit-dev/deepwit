package deepwit.init

import dimwit.*
import dimwit.stats.Normal
import dimwit.stats.Uniform
import dimwit.Label as Λ

/** Xavier/Glorot initializers, drawing weights with variance `2 / (fanIn + fanOut)` as described in
  * [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html).
  */
object Init:

  def xavierNormal[FanIn: Λ, FanOut: Λ, V: IsFloating](fanIn: AxisExtent[FanIn], fanOut: AxisExtent[FanOut], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1f): Tensor2[FanIn, FanOut, V] =
    val variance = Tensor0(vtype)(2.0f / (fanIn.size + fanOut.size))
    Normal.standardIsotropic(Shape(fanIn, fanOut), scale = gain * variance.sqrt).sample(key)

  def xavierUniform[FanIn: Λ, FanOut: Λ, V: IsFloating](fanIn: AxisExtent[FanIn], fanOut: AxisExtent[FanOut], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1f): Tensor2[FanIn, FanOut, V] =
    val variance = Tensor0(vtype)(2.0f / (fanIn.size + fanOut.size))
    // A uniform on [-a, a] has variance a²/3, so a = √(3·variance) hits the target spread.
    val a = gain * (3f * variance).sqrt
    IndependentDistribution.fromUnivariate(Shape(fanIn, fanOut), Uniform(-a, a)).sample(key)

  def xavierNormalVector[FanIn: Λ, V: IsFloating](fanIn: AxisExtent[FanIn], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1f): Tensor1[FanIn, V] =
    val fanOut = 1 // linear form has fan-out of 1
    val variance = Tensor0(vtype)(2.0f / (fanIn.size + fanOut))
    Normal.standardIsotropic(Shape(fanIn), scale = gain * variance.sqrt).sample(key)

  def xavierUniformVector[FanIn: Λ, V: IsFloating](fanIn: AxisExtent[FanIn], key: Key, vtype: VType[V] = VType[Float32], gain: Float = 1f): Tensor1[FanIn, V] =
    val fanOut = 1 // linear form has fan-out of 1
    val variance = Tensor0(vtype)(2.0f / (fanIn.size + fanOut))
    // A uniform on [-a, a] has variance a²/3, so a = √(3·variance) hits the target spread.
    val a = gain * (3f * variance).sqrt
    IndependentDistribution.fromUnivariate(Shape(fanIn), Uniform(-a, a)).sample(key)
