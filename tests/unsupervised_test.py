from causal_rl import model, train, tracker, config
from causal_rl.environments import nri_envs


def test_gumbel_vae():
    num_obj = 5
    obj_dim = 4
    m = model.GumbelVAE(num_obj, obj_dim, temp=0.5).to(config.DEVICE)
    train_loader = nri_envs.HardSpheres().next_state_dataloader(100, 1, 64, 1)
    writer = tracker.ConsoleTracker()
    trainer = train.TrainGumbel(m)
    trainer.train(train_loader, writer)
