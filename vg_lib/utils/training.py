import jax.numpy as jnp
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import optax
import flax.linen as nn

NUM_AGENTS = 8
AGENT_KEYS = [f'agent_{i}' for i in range(NUM_AGENTS)]
AGENT_OBS_KEYS = ['proprio','image']
CENTR_OBS_KEYS = ['wealth','has_resource','has_trash']
AGENT_AREA_KEYS = ['ecology_score','num_trash','num_resource','dead_ecology']

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, num_actors):
    x = jnp.stack([x[a] for a in AGENT_KEYS])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(AGENT_KEYS)}

def linear_schedule(count, config):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

def create_train_state(module: nn.Module, module_params: jnp.ndarray, config: dict[str, Any]) -> TrainState:
    tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
    train_state = TrainState.create(
            apply_fn=module.apply,
            params=module_params,
            tx=tx,
        )
    return train_state


