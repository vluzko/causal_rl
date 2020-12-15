# causal_rl

## Usage
`exp.py` contains scripts for running several different experiments. Output is stored in `plots` and `runs`.

## Installation
All tests are being performed with Python 3.8.5. You will need at least Python 3.6 to handle the type annotations.

You will need all the dependencies for [PyTorch](https://pytorch.org/get-started/locally/), particularly a CUDA-enabled GPU. The code *may* work on a CPU, but it will be very slow and it's untested.

* Run `pip install -r requirements.txt`
* Run `pip install .`
