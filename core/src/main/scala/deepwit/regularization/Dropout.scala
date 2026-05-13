package deepwit.regularization

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Bernoulli
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.embedder.VocabularyEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector

def sampleThinAffineLayer[In: Label, Out: Label, V: IsFloating](params: AffineLayer.Params[In, Out, V], dropoutRate: Float, key: Random.Key): AffineLayer.Params[In, Out, V] =
  val keepProb = 1.0f - dropoutRate
  val dropoutMask = IndependentDistribution.fromUnivariate(params.bias.shape, Bernoulli(Prob(keepProb))).sample(key).asFloat(VType[V]) *! (1f / (keepProb))
  params.copy(
    weight = params.weight *! dropoutMask,
    bias = params.bias * dropoutMask
  )

def sampleThinLinearLayer[L1: Label, L2: Label, V: IsFloating](params: LinearLayer.Params[L1, L2, V], dropoutRate: Float, key: Random.Key): LinearLayer.Params[L1, L2, V] =
  LinearLayer.Params(sampleThinProjection(params.weight, dropoutRate, key))

def sampleThinVocabularyEmbedder[Vocab: Label, Embedding: Label, V: IsFloating](params: VocabularyEmbedder.Params[Vocab, Embedding, V], dropoutRate: Float, key: Random.Key): VocabularyEmbedder.Params[Vocab, Embedding, V] =
  VocabularyEmbedder.Params(sampleThinProjection(params.vocabularyEmbeddings, dropoutRate, key))

def sampleThinLearnedAbsolutePositionalInjector[Context: Label, Embedding: Label, V: IsFloating](params: LearnedAbsolutePositionalInjector.Params[Context, Embedding, V], dropoutRate: Float, key: Random.Key): LearnedAbsolutePositionalInjector.Params[Context, Embedding, V] =
  LearnedAbsolutePositionalInjector.Params(sampleThinProjection(params.positionalEmbeddings, dropoutRate, key))

def sampleThinProjection[L1: Label, L2: Label, V: IsFloating](projMatrix: Tensor2[L1, L2, V], dropoutRate: Float, key: Random.Key): Tensor2[L1, L2, V] =
  val keepProb = 1.0f - dropoutRate
  val dropoutMask = IndependentDistribution.fromUnivariate(Shape1(projMatrix.shape.extent(Axis[L2])), Bernoulli(Prob(keepProb))).sample(key).asFloat(VType[V]) *! (1f / (keepProb))
  projMatrix *! dropoutMask
