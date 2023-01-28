from .causal_env import CausalEnv, TrivialEnv, ConstantEnv
from .dense_env import Torus
from .multi_typed import MultiTyped, WithTypes
from .mujoco import Mujoco
from .nri_envs import DenseEnv, HardSpheres


env_map = {
    "constant": ConstantEnv,
    "trivial": TrivialEnv,
    "dense": DenseEnv,
    "hard_spheres": HardSpheres,
}
