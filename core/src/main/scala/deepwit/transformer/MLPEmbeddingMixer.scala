package deepwit.transformer

import dimwit.*
import deepwit.activation.gelu
import deepwit.base.AffineLayer
import dimwit.Label as Λ

/** Mixes the components of a single embedding through a two-layer MLP.
  *
  * The embedding is expanded into the wider [[EmbeddingMixed]] space, passed through the
  * activation function, and projected back into the embedding space.
  *
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param activation The activation function applied to the expanded embedding.
  */
class MLPEmbeddingMixer[Embedding: Λ, V: IsFloating](
    params: MLPEmbeddingMixer.Params[Embedding, V],
    activation: Tensor1[EmbeddingMixed, V] => Tensor1[EmbeddingMixed, V]
) extends (Tensor1[Embedding, V] => Tensor1[Embedding, V]):

  private val expandLayer = AffineLayer(params.expand)
  private val projectLayer = AffineLayer(params.project)

  override def apply(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    val mixed = activation(expandLayer(embedding))
    projectLayer(mixed)

object MLPEmbeddingMixer:

  /** Defaults the activation to [[deepwit.activation.gelu]]. */
  def apply[Embedding: Λ, V: IsFloating](params: Params[Embedding, V]): MLPEmbeddingMixer[Embedding, V] =
    new MLPEmbeddingMixer(params, gelu)

  def apply[Embedding: Λ, V: IsFloating](
      params: Params[Embedding, V],
      activation: Tensor1[EmbeddingMixed, V] => Tensor1[EmbeddingMixed, V]
  ): MLPEmbeddingMixer[Embedding, V] =
    new MLPEmbeddingMixer(params, activation)

  case class Params[Embedding, V](
      expand: AffineLayer.Params[Embedding, EmbeddingMixed, V],
      project: AffineLayer.Params[EmbeddingMixed, Embedding, V]
  )

  object Params:

    def xavierNormal[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (expandKey, projectKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierNormal(embeddingExtent, embeddingMixedExtent, vtype, expandKey),
        project = AffineLayer.Params.xavierNormal(embeddingMixedExtent, embeddingExtent, vtype, projectKey)
      )

    def xavierUniform[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Random.Key): Params[Embedding, V] =
      val (expandKey, projectKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, expandKey),
        project = AffineLayer.Params.xavierUniform(embeddingMixedExtent, embeddingExtent, vtype, projectKey)
      )
