package example.gpt

import dimwit.*
import dimwit.Conversions.given
import deepwit.*
import deepwit.logging.TenZarrLogger
import Config.*
import nn.ActivationFunctions.softmax
import dimwit.FloatTree.ops.*

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import nn.Adam
import nn.AdamW
import nn.LearningRateSchedules.*

object BPETokenizer:
  private val tiktoken = py.module("tiktoken")
  private val enc = tiktoken.get_encoding("gpt2")

  def encode(text: String): Seq[Int] = enc.encode(text).as[Seq[Int]]

  def decode(tokens: Seq[Int]): String = enc.decode(tokens.toPythonCopy).as[String]

@main def generateText(): Unit =

  val promptText = "Here is my grandmother's secret recipe for the best chocolate chip cookies. Ingredients:"
  given key: Random.Key = Random.Key.fromTime()

  println("Loading model checkpoint...")

  val checkpointPath = "out/GPT-2/20260526_180211"
  val logger = new TenZarrLogger(checkpointPath)
  val initParams = GPT.Params.init(numTransformerLayers = numLayers)(
    vocabExtent,
    contextExtent,
    headExtent,
    headQueryExtent,
    headKeyExtent,
    headValueExtent,
    embeddingExtent,
    embeddingMixedExtent,
    VType[Float32],
    key
  )

  val schedule = linearWarmup(1e-4f, 1_000) min cosineDecay(1e-4f, 0.0f, 20_000).delay(1_000)
  val adamW = AdamW(
    Adam.withSchedule(schedule, b1 = 0.99f, b2 = 0.99f),
    weightDecayFactor = 0.1f
  )

  case class TrainingState(
      params: GPT.Params[Float32],
      adamWState: adamW.State[GPT.Params[Float32]],
      stepCost: Tensor0[Float32]
  )

  val dummyState = TrainingState(initParams, adamW.init(initParams), Tensor0(-1f))
  println("Load checkpoint!")
  val loadedState = new TenZarrLogger(f"out/GPT-2/20260526_065320").loadTensorTree[TrainingState](dummyState, "checkpoint", 18_006).get

  val hyperParams = GPT.HyperParams(Transformer.HyperParams(LayerNorm.HyperParams(1e-12)))

  val model = GPT(hyperParams)(loadedState.params.asFloats(VType[BFloat16]))

  println(s"\nPrompt: $promptText")
  print("Response: ")

  val promptTokens = BPETokenizer.encode(promptText)
  val tokenStream = model.generate(promptTokens, contextSize = contextExtent.size, temperature = 0.8f)
  val maxTokens = 200
  val eosTokenId = 50256

  tokenStream
    .takeWhile(_ != eosTokenId)
    .take(maxTokens)
    .foreach: tokenId =>
      val textChunk = BPETokenizer.decode(Seq(tokenId))
      print(textChunk)
      System.out.flush()

  println("\n\n[Generation Complete]")
