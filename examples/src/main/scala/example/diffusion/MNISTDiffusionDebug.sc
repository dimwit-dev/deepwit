import dimwit.*
import dimwit.Conversions.given
import deepwit.example.diffusion.*
import plotwit.*

dimwit.initialize()

val key = Random.Key(42)
val initialParams = DiffusionUNet.Params.xavierUniform(List(32, 64, 128))(key)

import plotwit.PlotTargets.desktopBrowser
display(plots.tensorTreeShapePlot(initialParams, _.title := "DiffusionUNet Parameters"))
