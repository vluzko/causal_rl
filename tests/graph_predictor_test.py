"""Tests for the graph predictor architectures.
These only test that they *run*, not that they *work*.
"""
from causal_rl.supervised.trainer import GraphTrainer, NextStateTrainer
from causal_rl.environments import nri_envs


def test_attention_predictor():
    """Test with an attentional graph predictor"""
    num_obj = 5
    env = nri_envs.DenseEnv(num_obj=num_obj)
    train_class = GraphTrainer(
        env,
        hidden_dim=128,
        num_layers=6,
        lr=0.00005,
        batch_size=128,
        num_obj=num_obj,
        length=100,
        num_train_sims=1,
        num_val_sims=1,
        tracker_string="console",
        attention=True,
    )
    train_class.train()
    train_class.validate()


def test_dense_net_predictor():
    """Test with a dense net predictor"""
    num_obj = 5
    env = nri_envs.DenseEnv(num_obj=num_obj)
    train_class = GraphTrainer(
        env,
        hidden_dim=128,
        num_layers=6,
        lr=0.00005,
        batch_size=128,
        num_obj=num_obj,
        length=100,
        num_train_sims=1,
        num_val_sims=1,
        tracker_string="console",
        attention=False,
    )
    train_class.train()
    train_class.validate()


def test_trivial():
    env = nri_envs.TrivialEnv()
    train_class = GraphTrainer(
        env,
        hidden_dim=128,
        num_layers=6,
        lr=0.00005,
        batch_size=128,
        num_obj=5,
        length=100,
        num_train_sims=1,
        num_val_sims=1,
        tracker_string="console",
        attention=False,
        upsample=False,
    )
    train_class.train()
    train_class.validate()


def test_convolution_predictor():
    num_obj = 5
    env = nri_envs.DenseEnv(num_obj=num_obj)
    train_class = GraphTrainer(
        env,
        hidden_dim=128,
        num_layers=6,
        lr=0.00005,
        batch_size=128,
        num_obj=num_obj,
        length=100,
        num_train_sims=1,
        num_val_sims=1,
        tracker_string="console",
        attention=True,
    )
    raise NotImplementedError
