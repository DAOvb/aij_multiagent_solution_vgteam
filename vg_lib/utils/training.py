import jax.numpy as jnp
import jax
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import optax
import flax.linen as nn

NUM_AGENTS = 8
AGENT_KEYS = [f"agent_{i}" for i in range(NUM_AGENTS)]
AGENT_OBS_KEYS = ["proprio", "image"]
CENTR_OBS_KEYS = ["wealth", "has_resource", "has_trash"]
AGENT_AREA_KEYS = ["ecology_score", "num_trash", "num_resource", "dead_ecology"]


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs_image: jnp.ndarray
    obs_feats: jnp.ndarray
    world_image: jnp.ndarray
    world_feats: jnp.ndarray
    info: jnp.ndarray


def batchify(x, key, config):
    x = jnp.stack([x[a][key] for a in AGENT_KEYS])
    return x.reshape(config["NUM_ACTORS"], -1)


def batchify_image(x, config):
    x = jnp.stack([x[a]["image"] for a in AGENT_KEYS])
    return x.reshape(config["NUM_ACTORS"], *x.shape[-3:])


def unbatchify_acts(x, num_envs):
    x = x.reshape((NUM_AGENTS, num_envs, -1))
    return {a: x[i].squeeze() for i, a in enumerate(AGENT_KEYS)}


def linear_schedule(count, config):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac


def create_train_state(
    module: nn.Module, module_params: jnp.ndarray, config: dict[str, Any]
) -> TrainState:
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


def process_world_state(state, area_state):
    image = state["image"]
    additional_obs = jnp.concatenate(
        [state[_] for _ in CENTR_OBS_KEYS], dtype=jnp.float32
    )
    area_obs = jnp.array(
        [
            [area_state[agent_key][_] for _ in AGENT_AREA_KEYS]
            for agent_key in AGENT_KEYS
        ],
        dtype=jnp.float32,
    ).reshape(-1)
    full_feats = jnp.concatenate([additional_obs, area_obs])
    return image, full_feats


def get_advantages(gae_and_next_value, transition, gamma, lmbd):
    gae, next_value = gae_and_next_value
    done, value, reward = (
        transition.global_done,
        transition.value,
        transition.reward,
    )
    delta = reward + gamma * next_value * (1 - done) - value
    gae = delta + gamma * lmbd * (1 - done) * gae
    return (gae, value), gae


def scan_adv(traj_batch, last_val, gamma, lmbd):
    _, advantages = jax.lax.scan(
        get_advantages,
        (jnp.zeros_like(last_val), last_val, gamma, lmbd),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value
