# DeepWit

This document illustrates the core concepts behind DeepWit, a deep learning library built on top of DimWit.

## Design Philosophy

We share DimWit's vision: **put human understanding first.** 
In the context of a deep learning library, this means breaking away from the tradition of hiding concepts behind opaque, internal state or side effects.
Instead, DeepWit forces the user to be more explicit, to bring the code into tight alignment with the underlying theory.
This transparency also has a highly practical benefit: when the foundational concepts are explicit, building advanced techniques on top of them becomes significantly easier to reason about.

### The Cost of Hidden Concepts

Many libraries hide core concepts internally to make user code more compact, but at the cost of transparency.
To understand what the code is doing and link it back to theory, a reader must memorize framework-specific details. 
This forces the user to learn two disconnected things: the actual theoretical concepts, and the implementation specifics.

To illustrate this disconnect between theory and code, consider PyTorch's approach to automatic differentiation (backprop):
```python
# Setup: We have some code calculating the loss
loss = ...
# (1) Run backprop and set gradients internally in tensor objects.
loss.backward()
# (2) Apply gradients inside the tensor objects to the parameter values.
optimizer.step()
```
Without prior knowledge of the framework, it is impossible to deduce what happens under the hood at (1) and (2). Looking solely at the syntax, there is no indication of what is being optimized, or even that gradient-based learning is occurring.

The equivalent in DeepWit looks like this:
```scala
// Setup: We have a function defining how to calculate the loss given parameters
def lossF(params: Params): Tensor0[Float] = ...
// (1) Run automatic differentiation to calculate the gradients.
val grads = Autodiff.grad(lossF)(params)
// (2) Apply gradients to parameters to get new parameters.
val nextParams = optimizer.update(grads, params)
```
Because we must explicitly represent the gradients and the parameters, the code naturally aligns with the theory of gradient-based learning.

Furthermore, having explicit gradients makes implementing custom training logic—like gradient accumulation—trivial. Instead of relying on additional, framework-specific workarounds, the user simply sums the explicit gradient values over multiple batches before applying them.

### The Beauty of Explicit Concepts

In DeepWit, we embrace **explicitness as a feature**, not a burden. While defining concepts explicitly requires slightly more upfront code, the return on investment is massive: it demystifies complex architectures and tightly couples the implementation back to the theory.

Breaking with the tradition of most existing deep learning frameworks, DeepWit requires the user to, among others, represent learnable parameters as explicit data objects, pass hyperparameters in a separate parameter group, and handle random effects manually throughout the architecture (see Core Concepts in DeepWit).
These design choices replace "framework magic" with a transparent implementation that remains highly aligned with the theory. In this directness, we find value and beauty.

## Core Concepts in DeepWit

This section describes some core concepts in DeepWit that significantly differ from established deep learning library in a beneficial way. 

### Explicit Parameter Representation

DeepWit requires parameters to be defined as dedicated **data objects**. While this adds a small amount of boilerplate, it brings the code into closer alignment with theory. It makes the implementation entirely transparent for critical tasks like weight initialization, updating parameters, augmentating parameters (e.g., dropout), and storing model weights (e.g., checkpointing).

For example, `AffineLayer.Params[In, Out]` represents the parameters for an affine layer. Because these are decoupled from the layer's logic, an explicit initialization method must be chosen (or implemented) _by the user_ at the time of creation.

```scala

object AffineLayer:

  case class Params[In, Out](
      weight: Tensor2[In, Out, Float],
      bias: Tensor1[Out, Float]
  )

  object Params:

    // Provide default, transparent strategies for initialization
    
    def xavierNormal[In: Label, Out: Label](
      inExtent: AxisExtent[In], outExtent: AxisExtent[Out], key: Random.Key
    ): Params[In, Out] = Params(
      weight = init.xavierNormal(inExtent, outExtent, key),
      bias = Tensor(Shape(outExtent)).fill(0f)
    )

    def xavierUniform(...) = ...

// Example of explicit parameters in user's code
val params = AffineLayer.Params.xavierDefault(Axis[Feature] -> 100, Axis[Output] -> 1)
val model = AffineLayer(params)

```

### Explicit Hyperparameter Group and Function

DeepWit modules follow a **curried constructor pattern** that requires hyperparameters first (if any), followed by parameters. This structural separation aligns the implementation with the theoretical transition from a model family to a mathematical function: Passing the hyperparameters fixes the "kind" of model, establishing its internal structure. Passing the parameters fixes the model's behavior to form a concrete mathematical function.

```scala
case class Model
  // 1: Define "kind" of model 
  (hyperParams: HyperParams)
  // 2: Define behavior of model
  (params: Params)
  // 3: Results in a concrete function
  extends (In => Out):
    override apply(in: In): Out = ...
```

### Explicit Randomness

DeepWit treats stochasticity as an explicit capability rather than a hidden side effect, utilizing the Random Key concept (cf. JAX). This ensures that the presence—and absence—of randomness is clearly visible in the parameters passed to a module, indicating conceptual determinism or stochasticity.

Most deep learning libraries—including PyTorch, TensorFlow, and even FLAX (built on top of JAX)—implement randomness in an implicit fashion by relying on effectful functions (e.g., torch.rand()) or global states (e.g., flax.nnx.Rngs). By breaking with this tradition, we ensure that a module's behavior is fully determined by its inputs. If a model is stochastic, its signature must say so.

## Downstream Benefits

By moving away from "framework magic" and leaning into intentional, DeepWit unlocks several powerful patterns. We find that when the library's structure aligns with the theory, the resulting user code isn't just more transparent—it is fundamentally more elegant and less error-prone. This section discusses some highlights. See the examples for the whole experience.

### Mathematically Aligned Loss Functions

DeepWit explicitly represents parameters as a data object, allowing one to define a loss function as it is mathematically defined $\mathcal{L} : \beta \rightarrow \mathbb{R}$, as a mapping from parameters $\beta$ to a scalar value.
Furthermore, we can define a more complete loss function $\mathcal{L}_{D} : \beta \rightarrow \mathbb{R}$ that is explicitly bound to a specific dataset $D$. This communicates that the loss function estimates empirical risk—an abstraction elegantly implemented using Scala 3 function currying.

```scala
def loss
  // (1) Provide dataset D to fix the data context (Empirical Risk)
  (inputs: X, targets: Y)
  // (2) Resulting in a function from parameters to scalar value
  (params: Model.Params): Tensor0[Float] = ...

// Defining specific loss functions becomes a natural consequence of the API
val trainLoss = loss(trainX, trainY)
val valLoss = loss(valX, valY)

// This "correctness" pays off during optimization:
val dTrainLoss = Autodiff.grad(trainLoss)
val grads = dTrainLoss(currentParams)
```

### TODO more highlights

TODO