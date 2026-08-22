# Thinning: dropout without a model that knows about it

DeepWit has no dropout layer. Dropout is implemented by **thinning** the network — deleting feature
directions from the *parameters*, between the optimizer and the loss, rather than from the
activations inside a forward pass. This example illustrates how that works.

## The problem with classic dropout

Dropout is regularization. It is a statement about *how you train*, not about *what the model is*.
Classic implementations say the opposite, putting dropout inside the model as a layer, which forces
a series of concessions:

- **The model knows about regularization.** A concern that belongs to the training loop leaks into
  the model's construction and its type, and the dropout rate is still sitting there at inference,
  where it means nothing.
- **The model grows modes.** `model.train()` and `model.eval()` are hidden mutable state that
  silently changes what the forward pass computes — two computation graphs for one model. Forgetting
  the switch is a classic quiet bug: the model still runs, just wrong.
- **A deterministic function turns stochastic.** The forward pass has to be handed a random key, or
  — worse — draws from a global generator, so the model is no longer a function of its inputs.

## Thinning the network instead of dropping activations

Classic dropout changes $f$. Each step splices a mask between the layers, producing a *different
network* $f_z$, and the loss is averaged over those networks. DeepWit changes $\theta$. The network
is one fixed function, and each step hands it a *different parameter set* $\theta_z$:

$$
\begin{array}{c|c}
\textbf{classic — a random network} & \textbf{thinning — random parameters} \\[6pt]
\displaystyle \min_\theta\ \mathbb{E}_z\big[\mathcal{L}(f_z(\theta))\big] &
\displaystyle \min_\theta\ \mathbb{E}_z\big[\mathcal{L}(f(\theta_z))\big]
\end{array}
$$

These are the same objective. To see it, look at one layer: a hidden feature vector $h$, and the
linear layer that reads it, $a = \theta^\top h$ — writing $\theta$ for that layer's weight matrix,
since thinning does the same to every layer that reads a thinned feature space.

Dropout deletes a random subset of $h$'s directions and rescales the survivors so that the mean is
unchanged. That is a diagonal matrix,

$$D = \frac{1}{1-p}\,\mathrm{diag}(z), \qquad z \sim \mathrm{Bernoulli}(1-p)$$

zero where a direction was deleted, $1/(1-p)$ where it survived. Apply it to the activation, or to
the weights:

$$\theta^\top (D\,h) \;=\; \theta^\top D\, h \;=\; (D\,\theta)^\top h \;=\; \underbrace{\theta_z^\top}_{\theta_z \,=\, D\theta}\, h$$

Only the bracketing changed, so $f(\theta_z) = f_z(\theta)$ for every draw — but the two sides say
something different about *when*. On the left, $D$ is applied while the layer computes, inside $f$.
On the right it was applied beforehand, in preparing $\theta_z$, and what the layer computes is an
ordinary $\theta_z^\top h$: the mask never enters the network at all.

No $\varphi$ appears anywhere in the argument either, because it has already been applied by the
time $h$ reaches the layer — so this holds whatever the activation function is.

The thinning goes *inside* the function being differentiated, so the chain rule puts that same $D$
on the gradient. Thinning a weight thins its derivative too: a deleted direction receives exactly
zero gradient, which is the whole of what the regularizer does.

```scala
val (value, grads) = Autodiff.valueAndGrad(
  (params: MoonsMLP.Params) => cost(params.thin(thinningProbability, thinningKey))
)(state.params)
```

## Conclusion

The model is an ordinary function of its parameters: no dropout rate in its type, no random key, and
no modes to switch. Training and inference run the very same computation graph — all that differs is
which parameters are handed to it. The optimizer steps the same tree it differentiated, so nothing
has to be undone afterwards and the checkpoint holds nothing but weights.
