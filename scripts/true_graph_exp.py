"""Training script for graph predictor + state predictor, written from scratch"""
from torch import optim
from torch.nn import functional
from fire import Fire

from causal_rl.environments import HardSpheres
from causal_rl.model import NaivePredictor
from causal_rl.state_predictor import WeightedPredictor
from causal_rl.config import DEVICE, TrackerKey
from causal_rl.utils import get_tracker


def naive(
    batch_size: int = 256,
    num_obj: int = 5,
    steps: int = 100000,
    lr: float = 1e-4,
    tracker_str: TrackerKey = "wandb",
):
    env = HardSpheres(num_obj=num_obj, use_cache=True)

    train_loader = env.generate_data(
        steps, num_process=2, batch_size=batch_size, num_sims=1
    )
    state_model = NaivePredictor(num_obj, 4).to(DEVICE)
    tracker = get_tracker(tracker_str, "w_true_graph", {"name": "No Graph"})

    opt = optim.Adam(state_model.parameters(), lr=lr)
    opt.zero_grad()
    for i, (batch, _, target) in enumerate(train_loader):
        prediction = state_model(batch)

        loss = functional.mse_loss(prediction, target)

        loss.backward()
        opt.step()
        opt.zero_grad()

        tracker.add_scalar("MSE", loss.item(), i)


def train(
    batch_size: int = 256,
    num_obj: int = 5,
    steps: int = 100000,
    lr: float = 1e-4,
    tracker_str: TrackerKey = "wandb",
):
    env = HardSpheres(num_obj=num_obj, use_cache=True)

    train_loader = env.generate_data(
        steps, num_process=2, batch_size=batch_size, num_sims=1
    )
    state_model = WeightedPredictor(num_obj, 4).to(DEVICE)
    tracker = get_tracker(tracker_str, "w_true_graph", {"name": "With Graph"})

    opt = optim.Adam(state_model.parameters(), lr=lr)
    opt.zero_grad()
    for i, (batch, edges, target) in enumerate(train_loader):
        prediction = state_model(batch, edges)

        loss = functional.mse_loss(prediction, target)

        loss.backward()
        opt.step()
        opt.zero_grad()

        tracker.add_scalar("MSE", loss.item(), i)


def combined(
    batch_size: int = 256,
    num_obj: int = 5,
    steps: int = 100000,
    lr: float = 1e-4,
    tracker_str: TrackerKey = "wandb",
):
    env = HardSpheres(num_obj=num_obj, use_cache=True)

    train_loader = env.generate_data(
        steps, num_process=2, batch_size=batch_size, num_sims=1
    )
    state_model = WeightedPredictor(num_obj, 4).to(DEVICE)
    naive_model = NaivePredictor(num_obj, 4).to(DEVICE)
    tracker = get_tracker(tracker_str, "w_true_graph", {})

    opt = optim.Adam(state_model.parameters(), lr=lr)
    naive_opt = optim.Adam(naive_model.parameters(), lr=lr)
    opt.zero_grad()
    naive_opt.zero_grad()
    for i, (batch, edges, target) in enumerate(train_loader):
        prediction = state_model(batch, edges)
        naive_pred = naive_model(batch)

        loss = functional.mse_loss(prediction, target)
        naive_loss = functional.mse_loss(naive_pred, target)

        loss.backward()
        naive_loss.backward()
        opt.step()
        naive_opt.step()
        opt.zero_grad()
        naive_opt.zero_grad()

        tracker.add_scalar("True Graph MSE", loss.item(), i)
        tracker.add_scalar("Simple Predictor MSE", naive_loss.item(), i)


if __name__ == "__main__":
    Fire()
    # print('Starting')
    # train()
    # naive()
    # combined()
