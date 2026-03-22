package deepwit.regularization

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Bernoulli
import deepwit.base.{AffineLayer, LinearLayer}
import deepwit.embedder.VocabularyEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector

def sampleThinAffineLayer[In: Label, Out: Label](params: AffineLayer.Params[In, Out], dropoutRate: Float, key: Random.Key): AffineLayer.Params[In, Out] =
  val keepProb = 1.0f - dropoutRate
  val dropoutMask = IndependentDistribution.fromUnivariate(params.bias.shape, Bernoulli(Prob(keepProb))).sample(key).asFloat *! (1f / (keepProb))
  params.copy(
    weight = params.weight *! dropoutMask,
    bias = params.bias * dropoutMask
  )

def sampleThinLinearLayer[L1: Label, L2: Label](params: LinearLayer.Params[L1, L2], dropoutRate: Float, key: Random.Key): LinearLayer.Params[L1, L2] =
  LinearLayer.Params(sampleThinProjection(params.weight, dropoutRate, key))

def sampleThinVocabularyEmbedder[Vocab: Label, Embedding: Label](params: VocabularyEmbedder.Params[Vocab, Embedding], dropoutRate: Float, key: Random.Key): VocabularyEmbedder.Params[Vocab, Embedding] =
  VocabularyEmbedder.Params(sampleThinProjection(params.vocabularyEmbeddings, dropoutRate, key))

def sampleThinLearnedAbsolutePositionalInjector[Context: Label, Embedding: Label](params: LearnedAbsolutePositionalInjector.Params[Context, Embedding], dropoutRate: Float, key: Random.Key): LearnedAbsolutePositionalInjector.Params[Context, Embedding] =
  LearnedAbsolutePositionalInjector.Params(sampleThinProjection(params.positionalEmbeddings, dropoutRate, key))

def sampleThinProjection[L1: Label, L2: Label](projMatrix: Tensor2[L1, L2, Float], dropoutRate: Float, key: Random.Key): Tensor2[L1, L2, Float] =
  val keepProb = 1.0f - dropoutRate
  val dropoutMask = IndependentDistribution.fromUnivariate(Shape1(projMatrix.shape.extent(Axis[L2])), Bernoulli(Prob(keepProb))).sample(key).asFloat *! (1f / (keepProb))
  projMatrix *! dropoutMask
