package deepwit.transformer

import dimwit.*
import deepwit.base.gelu
import deepwit.base.AffineLayer
import dimwit.Label as Λ

case class MLPEmbeddingMixer[Embedding: Λ, V: IsFloating](
    params: MLPEmbeddingMixer.Params[Embedding, V]
) extends (Tensor1[Embedding, V] => Tensor1[Embedding, V]):

  private val hiddenLayer = AffineLayer(params.expand)
  private val outputLayer = AffineLayer(params.project)

  override def apply(in: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    val hidden = gelu(hiddenLayer(in))
    outputLayer(hidden)

object MLPEmbeddingMixer:

  trait EmbeddingMixed derives Label

  case class Params[Embedding, V](
      expand: AffineLayer.Params[Embedding, EmbeddingMixed, V],
      project: AffineLayer.Params[EmbeddingMixed, Embedding, V]
  )

  object Params:

    def xavierNormal[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (fcKey, projKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierNormal(embeddingExtent, embeddingMixedExtent, vtype, fcKey),
        project = AffineLayer.Params.xavierNormal(embeddingMixedExtent, embeddingExtent, vtype, projKey)
      )

    def xavierUniform[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (fcKey, projKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, fcKey),
        project = AffineLayer.Params.xavierUniform(embeddingMixedExtent, embeddingExtent, vtype, projKey)
      )
