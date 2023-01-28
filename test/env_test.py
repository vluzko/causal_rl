from causal_rl.environments import HardSpheres


def test_hard_spheres():
    num_obj = 2
    steps = 100
    env = HardSpheres(num_obj=num_obj)
    states, rewards = env.generate_data(epochs=steps)
    assert states.shape == (steps, num_obj, 4)
