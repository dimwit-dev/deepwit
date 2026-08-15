package deepwit.optimizer

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors
import org.scalatest.Inspectors.forAll

import deepwit.*
import dimwit.optimizer.GradientDescent

class LearningRateScheduleSuite extends AnyFunSpec with Matchers:

  describe("LearningRateScheduler"):
    it("basic"):
      def learningRateSchedule(step: Tensor0[Int32]): Tensor0[Float32] =
        if step.item <= 5 then Tensor0(0.0f)
        else Tensor0(0.1f)
      val optimizer = LearningRateScheduler(lr => GradientDescent(lr), learningRateSchedule)
      val x1 = optimizer.iterate(Tensor0(2.0f))(x => Grad(2 * (x + 1))).drop(5).next()
      x1.item shouldBe 2.0f +- 0.1
      val x2 = optimizer.iterate(Tensor0(2.0f))(x => Grad(2 * (x + 1))).drop(1000).next()
      x2.item shouldBe -1.0f +- 0.1f

  describe("LearningRateSchedule"):

    it("constant schedule"):
      val linearWarmup = ConstantLearningRate(learningRate = 0.1f)
      val expectedLearningRates = List.fill(10)(0.1f).toArray
      val actualLearningRates = (1 to expectedLearningRates.size).map(step => linearWarmup(Tensor0(step)).item).toArray
      expectedLearningRates.length shouldBe actualLearningRates.length
      forAll(actualLearningRates.zip(expectedLearningRates)):
        case (actual, expected) => actual shouldBe (expected +- 1e-6f)

    it("constant + cosine decay schedule"):
      val constant5 = ConstantLearningRate(learningRate = 0.1f, steps = 5)
      val cosineDecay = CosineDecay(
        from = 0.1f,
        to = 0.0f,
        decaySteps = 10
      )
      val schedule = constant5.followBy(cosineDecay)
      val expectedLearningRates = Array(0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.09755282f, 0.09045085f, 0.07938926f, 0.06545085f, 0.05f, 0.03454915f, 0.020610744f, 0.009549153f, 0.0024471737f, 0.0f)
      val actualLearningRates = (1 to expectedLearningRates.size).map(step => schedule(Tensor0(step)).item).toArray
      expectedLearningRates.length shouldBe actualLearningRates.length
      forAll(actualLearningRates.zip(expectedLearningRates)):
        case (actual, expected) => actual shouldBe (expected +- 1e-6f)

    it("linear warmup schedule"):
      val linearWarmup = LinearWarmup(
        to = 0.1f,
        warmupSteps = 4
      )
      val expectedLearningRates = Array(0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.1f, 0.1f)
      val actualLearningRates = (1 to expectedLearningRates.size).map(step => linearWarmup(Tensor0(step)).item).toArray
      expectedLearningRates.length shouldBe actualLearningRates.length
      forAll(actualLearningRates.zip(expectedLearningRates)):
        case (actual, expected) => actual shouldBe (expected +- 1e-6f)

    it("cosine decay schedule"):
      val cosineDecay = CosineDecay(
        from = 1.0f,
        to = 0.0f,
        decaySteps = 10
      )
      val expectedLearningRates = Array(1.0f, 0.97552824f, 0.9045085f, 0.7938926f, 0.6545085f, 0.5f, 0.3454915f, 0.20610744f, 0.09549153f, 0.024471737f, 0.0f, 0.0f, 0.0f)
      val actualLearningRates = (1 to expectedLearningRates.size).map(step => cosineDecay(Tensor0(step)).item).toArray
      expectedLearningRates.length shouldBe actualLearningRates.length
      forAll(actualLearningRates.zip(expectedLearningRates)):
        case (actual, expected) => actual shouldBe (expected +- 1e-6f)

    it("linear warmup + cosine decay schedule"):
      val baseLearningRate = 0.1f
      val linearWarmup = LinearWarmup(
        to = baseLearningRate,
        warmupSteps = 4
      )
      val cosineDecay = CosineDecay(
        from = baseLearningRate,
        to = 0.0f,
        decaySteps = 10
      )
      val schedule = linearWarmup.followBy(cosineDecay)
      val expectedLearningRates = Array(0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.09755282f, 0.09045085f, 0.07938926f, 0.06545085f, 0.05f, 0.03454915f, 0.020610744f, 0.009549153f, 0.0024471737f, 0.0f)
      val actualLearningRates = (1 to expectedLearningRates.size).map(step => schedule(Tensor0(step)).item).toArray
      expectedLearningRates.length shouldBe actualLearningRates.length
      forAll(actualLearningRates.zip(expectedLearningRates)):
        case (actual, expected) => actual shouldBe (expected +- 1e-6f)

    it("linear warmup + cosine decay + linear warmup + cosine decay schedule"):
      val baseLearningRate = 0.1f
      val linearWarmup = LinearWarmup(
        to = baseLearningRate,
        warmupSteps = 4
      )
      val cosineDecay = CosineDecay(
        from = baseLearningRate,
        to = 0.0f,
        decaySteps = 10
      )
      val schedule = linearWarmup.followBy(cosineDecay).followBy(linearWarmup).followBy(cosineDecay)
      val expectedLearningRates = Array(0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.09755282f, 0.09045085f, 0.07938926f, 0.06545085f, 0.05f, 0.03454915f, 0.020610744f, 0.009549153f, 0.0024471737f, 0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.09755282f, 0.09045085f, 0.07938926f, 0.06545085f, 0.05f, 0.03454915f, 0.020610744f, 0.009549153f, 0.0024471737f, 0.0f, 0.0f)
      val actualLearningRates = (1 to expectedLearningRates.size).map(step => schedule(Tensor0(step)).item).toArray
      expectedLearningRates.length shouldBe actualLearningRates.length
      forAll(actualLearningRates.zip(expectedLearningRates)):
        case (actual, expected) => actual shouldBe (expected +- 1e-6f)
