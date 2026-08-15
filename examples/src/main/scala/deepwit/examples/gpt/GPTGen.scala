package deepwit.examples.gpt

import dimwit.*
import Config.*
import dimwit.TreeOf.ops.*

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

object BPETokenizer:
  private val tiktoken = py.module("tiktoken")
  private val enc = tiktoken.get_encoding("gpt2")

  def encode(text: String): Seq[Int] = enc.encode(text).as[Seq[Int]]

  def decode(tokens: Seq[Int]): String = enc.decode(tokens.toPythonCopy).as[String]

@main def generateText(): Unit =

  val promptText = "Here is my grandmother's secret recipe for the best chocolate chip cookies. Ingredients:"
  given key: Key = Random.Key.fromTime()

  println("Loading model checkpoint...")
  val params: GPT.Params[Float32] = ???

  val model = GPT(params.asFloats(VType[BFloat16]))

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
