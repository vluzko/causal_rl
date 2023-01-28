"""Supervised causal graph predictor experiments."""
import torch
import numpy as np
import random

from fire import Fire
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from typing import Optional
from causal_rl.environments.causal_env import CausalEnv

from causal_rl.graph_predictor import DenseNetNoDiag
from causal_rl.supervised.trainer import GraphTrainer, NextStateTrainer
from causal_rl.environments import TrivialEnv, ConstantEnv
from causal_rl.environments.nri_envs import (
    HardSpheres,
    DenseEnv,
)
from causal_rl.config import TrackerKey


def hyper_opt():
    # Grid search
    # analysis = tune.run(
    #     asym_dense,
    #     config = {
    #         "lr": tune.grid_search([0.00005, 0.0005, 0.005])
    #     }
    # )

    def wrapper(config):
        # seriously raytune
        return asym_dense(**config)

    current_best_params = [{"lr": 0.0005}]

    hopt_alg = HyperOptSearch(points_to_evaluate=current_best_params)
    concurrent = ConcurrencyLimiter(hopt_alg, max_concurrent=4)
    analysis = tune.run(
        wrapper,
        search_alg=concurrent,
        mode="min",
        num_samples=10,
        config={
            "lr": tune.uniform(0.000005, 0.005),
        },
        resources_per_trial={"gpu": 1},
    )


def benchmark_nextstate():
    train_class = NextStateTrainer("d2rl", 0.0001, 64, 5, 50000, 1, 1)
    train_class.train()
    train_class.validate()


def dense_predictor(num_obj: int = 5, tracker_string: TrackerKey = "wandb"):
    env = TrivialEnv()
    train_class = GraphTrainer(
        env,
        hidden_dim=128,
        num_layers=6,
        lr=0.00005,
        batch_size=128,
        num_obj=num_obj,
        length=50000,
        num_train_sims=1,
        num_val_sims=1,
        tracker_string=tracker_string,
        attention=False,
    )
    train_class.train()
    train_class.validate()


def asym_dense(
    num_obj: int = 5,
    tracker_string: TrackerKey = "wandb",
    env_name: str = "hard_spheres",
    lr: float = 0.0005,
) -> float:
    env: CausalEnv
    if env_name == "dense":
        env = DenseEnv(num_obj=num_obj, use_cache=True)
    elif env_name == "trivial":
        env = TrivialEnv()
    elif env_name == "constant":
        env = ConstantEnv()
    elif env_name == "hard_spheres":
        env = HardSpheres(num_obj=num_obj, use_cache=True)
    else:
        raise ValueError(env_name)
    model_cls = DenseNetNoDiag
    train_class = GraphTrainer(
        env,
        hidden_dim=256,
        num_layers=6,
        lr=lr,
        batch_size=256,
        num_obj=num_obj,
        length=20000,
        num_train_sims=1,
        num_val_sims=1,
        tracker_string=tracker_string,
        attention=False,
        model_cls=model_cls,
    )
    train_class.train()
    return 0.0
    # return train_class.validate()


def attention_predictor(
    num_obj: int = 5,
    env_name: str = "trivial",
    upsample: bool = False,
    tracker: TrackerKey = "wandb",
    seed: Optional[int] = 1,
):
    env: CausalEnv
    if env_name == "dense":
        env = DenseEnv(num_obj=num_obj)
    elif env_name == "trivial":
        env = TrivialEnv()
    elif env_name == "hard_spheres":
        env = HardSpheres(num_obj=num_obj)
    else:
        raise ValueError(env_name)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    train_class = GraphTrainer(
        env,
        hidden_dim=128,
        num_layers=6,
        lr=0.01,
        batch_size=128,
        num_obj=5,
        length=100000,
        num_train_sims=1,
        num_val_sims=1,
        tracker_string=tracker,
        attention=False,
        sigmoid=False,
        upsample=upsample,
    )
    train_class.train()
    train_class.validate()


if __name__ == "__main__":
    Fire()
