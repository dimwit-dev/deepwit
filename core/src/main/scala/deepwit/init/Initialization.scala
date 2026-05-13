package deepwit.init

import dimwit.*
import dimwit.stats.Normal
import dimwit.stats.Uniform

def xavierNormal[FanIn: Label, FanOut: Label, V: IsFloating](fanIn: AxisExtent[FanIn], fanOut: AxisExtent[FanOut], vtype: VType[V], key: Random.Key, gain: Float = 1f): Tensor2[FanIn, FanOut, V] =
  val variance = Tensor0(vtype)(2.0f / (fanIn.size + fanOut.size))
  Normal.standardIsotropic(Shape(fanIn, fanOut), scale = gain * variance.sqrt).sample(key)

def xavierUniform[FanIn: Label, FanOut: Label, V: IsFloating](fanIn: AxisExtent[FanIn], fanOut: AxisExtent[FanOut], vtype: VType[V], key: Random.Key, gain: Float = 1f): Tensor2[FanIn, FanOut, V] =
  val variance = Tensor0(vtype)(2.0f / (fanIn.size + fanOut.size))
  val a = gain * (3f * variance).sqrt
  IndependentDistribution.fromUnivariate(Shape(fanIn, fanOut), Uniform(-a, a)).sample(key)
