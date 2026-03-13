package deepwit.transformer

import dimwit.*
import deepwit.base.ActivationFunction.gelu
import deepwit.base.AffineLayer

case class MLPEmbeddingMixer[Embedding: Label](
    hyperParams: MLPEmbeddingMixer.HyperParams[Embedding]
)(
    params: MLPEmbeddingMixer.Params[Embedding]
) extends (Tensor1[Embedding, Float] => Tensor1[Embedding, Float]):

  private val hiddenLayer = AffineLayer(params.expand)
  private val outputLayer = AffineLayer(params.project)

  override def apply(in: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
    val hidden = gelu(hiddenLayer(in))
    outputLayer(hidden)

object MLPEmbeddingMixer:

  trait EmbeddingMixed derives Label

  case class HyperParams[Embedding](
      activationFunction: Tensor[Tuple1[Embedding], Float] => Tensor[Tuple1[Embedding], Float]
  )

  case class Params[Embedding](
      expand: AffineLayer.Params[Embedding, EmbeddingMixed],
      project: AffineLayer.Params[EmbeddingMixed, Embedding]
  )

  object Params:

    def xavierNormal[Embedding: Label](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Random.Key): Params[Embedding] =
      val (fcKey, projKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierNormal(embeddingExtent, embeddingMixedExtent, fcKey),
        project = AffineLayer.Params.xavierNormal(embeddingMixedExtent, embeddingExtent, projKey)
      )

    def xavierUniform[Embedding: Label](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Random.Key): Params[Embedding] =
      val (fcKey, projKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, fcKey),
        project = AffineLayer.Params.xavierUniform(embeddingMixedExtent, embeddingExtent, projKey)
      )
