from vg_lib.env.vec_env import create_vec_env, VectorParallelEnv
from config import get_config
from aij_multiagent_rl.env import AijMultiagentEnv
from vg_lib.env.framestack import FrameStack


def _create_env():
    return FrameStack(AijMultiagentEnv())


def make_training_env() -> VectorParallelEnv:
    config = get_config()

    vec_env = create_vec_env(
        _create_env,
        config.num_agents)

    return vec_env