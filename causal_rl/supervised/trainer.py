import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from typing import Optional, Type

from causal_rl.graph_predictor import (
    CDAGPredictor,
    MLPBase,
    D2RLNet,
    DenseNetGraphPredictor,
    AttentionGraphPredictor,
)
from causal_rl.environments.nri_envs import HardSpheres
from causal_rl import utils, config


class GraphTrainer:
    def __init__(
        self,
        env,
        hidden_dim: int,
        num_layers: int,
        lr: float,
        batch_size: int,
        num_obj: int,
        length: int,
        num_train_sims: int,
        num_val_sims: int,
        tracker_string: config.TrackerKey = "console",
        attention: bool = False,
        sigmoid: bool = True,
        upsample: bool = False,
        model_cls: Optional[Type[CDAGPredictor]] = None,
    ):
        self.batch_size = batch_size
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tracker = utils.get_tracker(
            tracker_string,
            "supervised_graph",
            {
                "hidden_dim": hidden_dim,
                "lr": lr,
                "num_layers": num_layers,
                "sigmoid": sigmoid,
                "upsample": upsample,
                "name": "20 Particles",
            },
        )  # TODO: make this better
        self.environment = env
        self.train_loader = self.environment.graph_predictor_dataloader(
            length=length,
            num_sims=num_val_sims,
            batch_size=batch_size,
            num_process=32,
            set_max_min=True,
            attention=attention,
            upsample=upsample,
        )
        self.sigmoid = sigmoid
        if model_cls is not None:
            self.model = model_cls(
                self.environment.num_obj,
                self.environment.obj_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        elif attention:
            self.model = AttentionGraphPredictor(
                self.environment.num_obj,
                self.environment.obj_dim,
                dim_head=hidden_dim,
                num_layers=num_layers,
                sigmoid=sigmoid,
            )  # TODO: make this better
        else:
            self.model = DenseNetGraphPredictor(
                self.environment.num_obj,
                self.environment.obj_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                sigmoid=sigmoid,
            )
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)

    def train(self):
        self.model.train()
        self.opt.zero_grad()
        for i, (feature, graph) in enumerate(self.train_loader):
            feature = feature.to(self.device)
            graph = graph.to(self.device)
            predicted_graph = self.model(feature)

            # if self.sigmoid:
            loss = F.binary_cross_entropy(predicted_graph, graph)
            # else:
            # loss = F.mse_loss(predicted_graph, graph)
            loss.backward()
            self.opt.step()
            # self.tracker.add_scalar('grad_mean', self.model.dense_net.output.weight.grad.sum(dim=1).abs().mean().item(), i)
            self.opt.zero_grad()

            # if self.sigmoid:
            self.tracker.add_scalar("cross_entropy", loss.item(), i)
            # else:
            # self.tracker.add_scalar("mse", loss.item(), i)
            # total_nonzero = graph.count_nonzero(dim=1).sum(dim=1)

            # self.tracker.add_scalar("fraction nonzero", total_nonzero.float().mean() / graph[0].numel(), i)

    def validate(self) -> float:
        self.model.eval()
        baseline = torch.zeros(self.environment.num_obj, self.environment.num_obj)
        losses = []
        trivial_losses = []
        for inputs, target in self.valid_loader:
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            prediction = self.model(inputs)
            baseline_batch = torch.cat(target.shape[0] * [baseline.unsqueeze(0)])
            baseline_batch = baseline_batch.to(self.device)
            if self.sigmoid:
                loss = F.binary_cross_entropy(prediction, target)
                trivial_loss = F.binary_cross_entropy(baseline_batch, target)
            else:
                loss = F.mse_loss(prediction, target)
                trivial_loss = F.mse_loss(baseline_batch, target)
            losses.append(loss.item())
            trivial_losses.append(trivial_loss.item())
        self.tracker.set_summary("val_loss_avg", np.mean(losses))
        self.tracker.set_summary("val_loss_trivial", np.mean(trivial_losses))
        return np.mean(losses)


class NextStateTrainer:
    def __init__(
        self,
        model_name: str,
        lr: float,
        batch_size: int,
        num_obj: int,
        length: int,
        num_train_sims: int,
        num_val_sims: int,
        tracker_string: config.TrackerKey = "console",
    ):
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tracker = utils.get_tracker(
            tracker_string, "next_state", {"network": model_name, "lr": lr}
        )  # make this better
        self.environment = HardSpheres(num_obj=num_obj)
        self.train_loader = self.environment.next_state_dataloader(
            length=length,
            num_sims=num_train_sims,
            batch_size=batch_size,
            num_process=32,
            set_max_min=True,
        )
        self.valid_loader = self.environment.next_state_dataloader(
            length=length,
            num_sims=num_val_sims,
            batch_size=batch_size,
            num_process=32,
            set_max_min=False,
        )
        if model_name == "mlp":  # fix this
            self.model = MLPBase(
                num_obj * self.environment.obj_dim, num_obj * self.environment.obj_dim
            )
        elif model_name == "d2rl":
            self.model = D2RLNet(
                num_obj * self.environment.obj_dim, num_obj * self.environment.obj_dim
            )
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)

    def train(self):
        self.model.train()
        self.opt.zero_grad()
        i = 0
        for batch in self.train_loader:
            input, target = batch
            input = input.to(self.device)
            target = target.to(self.device)
            prediction = self.model(input)

            loss = F.mse_loss(prediction, target)

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            self.tracker.add_scalar("mse", loss.item(), i)
            i += 1

    def validate(self):
        self.model.eval()
        losses = []
        for batch in self.valid_loader:
            input, target = batch
            input = input.to(self.device)
            target = target.to(self.device)
            prediction = self.model(input)

            loss = F.mse_loss(prediction, target)
            losses.append(loss.item())
        self.tracker.set_summary("val_loss_avg", np.mean(losses))
