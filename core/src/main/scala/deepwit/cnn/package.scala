package deepwit

import dimwit.tensor.AxisExtent

package object cnn:
  // 2D Convolutions
  export deepwit.cnn.`2d`.{AffineConv2DLayer, LinearConv2DLayer, TransposeAffineConv2DLayer, TransposeLinearConv2DLayer}
