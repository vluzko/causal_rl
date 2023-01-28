from causal_rl.environments import HardSpheres


def test_hard_spheres():
    num_obj = 2
    steps = 100
    env = HardSpheres(num_obj=num_obj)
    states, rewards = env.generate_data(epochs=steps)
    assert states.shape == (steps, num_obj, 4)


def test_nri_hard_spheres():
    raise NotImplementedError


def test_dense_env():
    env = Torus(5)
    data = env.generate_data(100)
