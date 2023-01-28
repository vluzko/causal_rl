"""Tests for the next state predictor architectures.
These only test that they *run*, not that they *work*
"""
from causal_rl.supervised.trainer import GraphTrainer, NextStateTrainer
from causal_rl.environments.nri_envs import HardSpheres, DenseEnv


def test_next_state():
    train_class = NextStateTrainer("d2rl", 0.0001, 64, 5, 50000, 1, 1)
    train_class.train()
    train_class.validate()
