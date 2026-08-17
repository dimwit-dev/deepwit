package deepwit.training

import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class MonitorSuite extends AnyFunSpec with Matchers:

  import Monitor.{ConcatMonitor, LearningRateMonitor, LossMonitor, PerformanceMonitor, StepMonitor, ThroughputMonitor}
  import deepwit.optimizer.LearningRateSchedules.ConstantLearningRate
  import dimwit.Conversions.given

  private case class State(loss: Float)

  describe("StepMonitor"):

    it("reports the step"):
      StepMonitor[State]().report(7, State(1f)) shouldBe "Step 7"

  describe("LossMonitor"):

    it("reports the loss with four decimals"):
      LossMonitor[State](_.loss).report(0, State(0.123456f)) shouldBe "Loss: 0.1235"

  describe("ConcatMonitor"):

    it("joins its monitors with the separator"):
      val monitor = ConcatMonitor[State](List(StepMonitor(), LossMonitor(_.loss)), sep = " | ")
      monitor.report(3, State(2f)) shouldBe "Step 3 | Loss: 2.0000"

    it("reports an empty string when it has no monitors"):
      ConcatMonitor[State](List.empty).report(3, State(2f)) shouldBe ""

  describe("ThroughputMonitor"):

    it("labels the throughput with the given unit"):
      ThroughputMonitor[State](unitsPerStep = 4, unitName = "tokens").report(1, State(1f)) should endWith("tokens/sec")

    it("rejects a decreasing step"):
      val monitor = ThroughputMonitor[State](unitsPerStep = 4, unitName = "tokens")
      monitor.report(10, State(1f))
      an[IllegalArgumentException] should be thrownBy monitor.report(9, State(1f))

  describe("PerformanceMonitor"):

    it("reports sample throughput"):
      PerformanceMonitor[State](batchSize = 4).report(1, State(1f)) should endWith("samples/sec")

  describe("LearningRateMonitor"):

    it("reports the scheduled learning rate in exponential notation"):
      LearningRateMonitor[State](ConstantLearningRate(1e-3f)).report(7, State(1f)) shouldBe "LR: 1.00e-03"

  describe("Monitor.default"):

    it("combines step, loss and throughput"):
      val monitor = Monitor.default[State](batchSize = 8, lossLens = _.loss)
      val report = monitor.report(2, State(0.5f))
      report should include("Step 2")
      report should include("Loss: 0.5000")
      report should include("samples/sec")
