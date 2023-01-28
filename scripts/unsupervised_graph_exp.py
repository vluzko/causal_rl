from fire import Fire

from causal_rl import model, train, tracker, config, graph_predictor, utils
from causal_rl.environments import env_map
from causal_rl.environments.nri_envs import HardSpheres


def gumbel(
    steps: int = 20000,
    tracker_str: config.TrackerKey = "wandb",
    batch_size: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 4,
    env_name: str = "hard_spheres",
):
    # env_class = env_map[env_name]
    env = HardSpheres(5, use_cache=True)
    # env = env_class(use_cache=True)
    gp = graph_predictor.DenseNetNoDiag(
        env.num_obj, env.obj_dim, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(config.DEVICE)
    m = model.GumbelVAE(env.num_obj, env.obj_dim, gp, temp=0.5, hard=False).to(
        config.DEVICE
    )
    # train_loader = env.generate_data(steps, batch_size=batch_size)
    train_loader = env.next_state_dataloader(steps, 1, batch_size, 1, set_max_min=True)

    writer = utils.get_tracker(tracker_str, "unsupervised", {})
    # writer: tracker.Tracker
    # if tracker_str == "console":
    #     writer = tracker.ConsoleTracker()
    # elif tracker_str == "wandb":
    #     writer = tracker.WandBTracker('')
    # else:
    #     raise ValueError(tracker_str)
    trainer = train.TrainGumbel(m)
    trainer.train(train_loader, writer)


def phys(
    num_obj: int = 5,
    steps: int = 20000,
    tracker_str: config.TrackerKey = "console",
    batch_size: int = 512,
    hidden_dim: int = 512,
    num_layers: int = 6,
    env_name: str = "hard_spheres",
):
    env_class = env_map[env_name]
    env = env_class(num_obj)
    gp = graph_predictor.DenseNetNoDiag(
        env.num_obj, env.obj_dim, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(config.DEVICE)
    m = model.GumbelVAE(env.num_obj, env.obj_dim, gp, temp=0.5, hard=False).to(
        config.DEVICE
    )
    train_loader = env.next_state_dataloader(
        "gumbel_run", steps, 1, batch_size, 1, set_max_min=True, attention=True
    )

    writer: tracker.Tracker
    if tracker_str == "console":
        writer = tracker.ConsoleTracker()
    elif tracker_str == "wandb":
        writer = tracker.WandBTracker()
    else:
        raise ValueError(tracker_str)
    trainer = train.TrainGumbel(m)
    trainer.train(train_loader, writer)
    # raise NotImplementedError


if __name__ == "__main__":
    Fire()
