package deepwit.init

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal
import dimwit.stats.Uniform

def xavierNormal[FanIn: Label, FanOut: Label](fanIn: AxisExtent[FanIn], fanOut: AxisExtent[FanOut], key: Random.Key, gain: Float = 1f): Tensor2[FanIn, FanOut, Float] =
  val variance = Tensor0(2.0f / (fanIn.size + fanOut.size))
  Normal.standardIsotropic(Shape(fanIn, fanOut), scale = gain * variance.sqrt).sample(key)

def xavierUniform[FanIn: Label, FanOut: Label](fanIn: AxisExtent[FanIn], fanOut: AxisExtent[FanOut], key: Random.Key, gain: Float = 1f): Tensor2[FanIn, FanOut, Float] =
  val variance = Tensor0(2.0f / (fanIn.size + fanOut.size))
  val a = gain * (3f * variance).sqrt
  IndependentDistribution.fromUnivariate(Shape(fanIn, fanOut), Uniform(-a, a)).sample(key)
