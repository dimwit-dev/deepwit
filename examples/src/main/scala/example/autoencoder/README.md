# DeepWit Example: AutoEncoder

This example follows DeepWit's Train-Trajectory Paradigm

## The Train-Trajectory Paradigm

There are three distinct steps in Deep Learning:
1. Define the model as a case class that takes optionally hyperparameters, then parameters and is itself a function from inputs to outputs. So it has the following form `HyperParams => Params => Function` or `Params => Function` in case of no hyperparameters.
2. Train the model to find a point-estimate `params: Params`.
3. Evaluate and Understand: Evaluate the specific model (`model(params)`) on its downstream performance, and if necessary, run experiments with the model to understand it better.

In DeepWit's Train-Trajectory Paradigm the code clearly separates these steps.
This breaks with current Deep Learning paradigms where 2., and 3. are often intertwined for example by logging intermediate results to Weights&Biases or TensorBoard. In DeepWit the training script only does iterative parameter optimization, while logging checkpoints. Evaluation and Understanding (e.g., Visualization) happens separately based on checkpoints.

### Motivation

The following things motivate separating evaluation from training:
1. `Reproducability`: All evaluation results and visualizations for understanding must be reproducable (given a checkpoint).
2. `Iterative nature`: Due to iterative optimization the model gets iteratively better. So, next to evaluating the final model, evaluating intermediate steps is very insightful about training dynamics. 
3. `Creative process`: Evaluation and visualization is a creative process, where results inform next steps, for example, visualizing latent states may motivate new visualizations.

### Method

In DimWit the training of a model is exactly two steps:
1. Setup a LazyList over a training state that represents the training trajectory through this state space (including parameters, learning rate schedule, etc.)
2. Run this trajectory while storing intermediate results, i.e., checkpoints. Report minor training debug information such as train loss, runtime performance metrics, these should add no overhead. Do **not** report complex evaluation like plotting example inputs and results, autoregressive predictions, etc.


