package deepwit.transformer

import dimwit.*

def causalMask[Context: Label, CrossContext: Label](
    scoreShape: Shape2[Context, CrossContext]
): Tensor[(Context, CrossContext), Bool] =
  tril(fullMask(scoreShape))

def fullMask[Context: Label, CrossContext: Label](
    scoreShape: Shape2[Context, CrossContext]
): Tensor[(Context, CrossContext), Bool] =
  Tensor(scoreShape).fill(true)
