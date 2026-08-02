import dimwit.*
import dimwit.Conversions.given
import deepwit.logging.TensorTreeLogger
import deepwit.example.mnist_classification.*

import plotwit.plotting.Plotting
import plotwit.plotting.Plotting.grid

import viz.PlotTargets.desktopBrowser

dimwit.initialize()

val logger = new TensorTreeLogger("/Users/mebr/Documents/Privat/Projects/deepwit/examples/out/MNistCNN/20260802_161901")
val state = logger.load[TrainState](2000).get
val currentModel = MNistCNN(state.params)

def plotKernels[In: Label, Out: Label](kernels: Tensor[(Height, Width, In, Out), Float32]): Unit =
  val allKernels = kernels.flatten((Axis[In], Axis[Out])).unstack(Axis[In |*| Out])
  // val (min, max) = (kernels.min, kernels.max)
  val specs = allKernels.zipWithIndex.map: (kernel, index) =>
    val (min, max) = (kernel.min, kernel.max)
    val kernelImg = (((kernel -! min) /! (max - min)) *! 255f).asInt(VType[UInt8])
    Plotting.image.plot(
      kernelImg,
      _.title := f"$index [${min.item}%.2f, ${max.item}%.2f]",
      _.mark.width := 100,
      _.mark.height := 100
    )
  Plotting.display(grid(specs.grouped(8).toSeq))

plotKernels(state.params.conv1.kernel)
plotKernels(state.params.conv2.kernel)
