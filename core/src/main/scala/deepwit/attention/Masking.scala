package deepwit.attention

import dimwit.*
import dimwit.Label as Λ

def causalMask[Context: Λ, CrossContext: Λ](
    scoreShape: Shape2[Context, CrossContext]
): Tensor[(Context, CrossContext), Bool] =
  tril(fullMask(scoreShape))

def fullMask[Context: Λ, CrossContext: Λ](
    scoreShape: Shape2[Context, CrossContext]
): Tensor[(Context, CrossContext), Bool] =
  Tensor(scoreShape).fill(true)
