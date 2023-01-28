# causal_rl

## Overview
Overall goal: train MBRL agents that have causal models of environments. The code is split into three parts:
- The environments (`environments/`). The key point is that these environments are annotated with their causal graphs, making it possible to assess performance at all
- The state models (`graph_predictor.py`, `state_predictor.py`, `model.py`). These predict the current causal graph (from state) and the next state (from current state and graph). `models` is a wrapper around both.
- MBRL algorithms. Totally unimplemented, because no environment is both causal and RL yet. (Note: probably CausalWorld solves this).

`exp.py` contains leftover code from my thesis. It's in the process of being refactored, but for now it contains examples of running the full code.

Data goes in `data/`. Logs go in `logs/`. Plots go in `plots/`. Most likely this will change and logs/plots will get handled by wandb or a similar service.

### Architecture
- Environments: Produce a state, *and* a causal graph associated with that state. The causal graph is *not* returned to the learner but does get used for scoring after the fact (i.e. it's not part of the training loop)
- Models: Are composed of two parts. The first (the graph predictor) takes state as input and produces a prediction of the causal graph. The second (the state predictor) takes state and the predicted causal graph as input, and outputs the next state.
    - Graph predictors are maps from $reals^k -> reals^(k^2)$. Entry $i, j$ of the result matrix is the probability that $state[i]$ at the current time step has a causal influence on $state[j]$ of the next time step
    - State predictors are maps from $reals^k x reals^(k^2) -> reals^k$.


## Usage
`exp.py` contains scripts for running several different experiments. Output is stored in `plots` and `runs`.

## Installation
All tests are being performed with Python 3.8.5. You will need at least Python 3.6 to handle the type annotations.

You will need all the dependencies for [PyTorch](https://pytorch.org/get-started/locally/), particularly a CUDA-enabled GPU. The code *may* work on a CPU, but it will be very slow and it's untested.

* Run `pip install -r requirements.txt`
* Run `pip install .`
