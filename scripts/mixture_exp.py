import torch
import numpy as np

from matplotlib import pyplot as plt

from causal_rl import mixture_model, environments, model

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def plot(losses):
    fig, ax = plt.subplots()

    ax.plot(losses, linestyle="None", marker="o", markersize=1)

    upper = np.ceil(np.log10(losses.max()))
    ax.set_ylim([0, 10**upper])
    ax.set_xlabel("Step")
    ax.set_ylabel("L2 Loss")
    plt.show()


def train():
    env = environments.HardSpheres()
    model_1 = model.GraphAndMessages(env.num_obj, env.obj_dim).to(DEVICE)
    model_2 = model.GraphAndMessages(env.num_obj, env.obj_dim).to(DEVICE)
    m = mixture_model.DualModel(env.num_obj * env.obj_dim, model_1, model_2)
    losses = mixture_model.run_dual(env, m, 1000)
    plot(losses)


train()
