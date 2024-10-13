
import collections
import copy
import functools
import typing as tp
from typing import Any, Callable, Dict, List

import einops
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
from vg_lib.utils.collate import smart_collate, smart_concat
from vg_lib.utils.gae import gae_advantages

from aij_multiagent_rl.agents import BaseAgent


ExpTuple = collections.namedtuple(
    'ExpTuple', ['state', 'action', 'reward', 'value', 'log_prob', 'done']
)


def encode_fourier_features(value, num_frequencies=8):
    """
    Encodes a scalar value into Fourier features.

    Args:
        value: The scalar value to encode.
        num_frequencies: The number of frequency components to use.

    Returns:
        A 1D JAX array of Fourier features.
    """
    frequencies = jnp.arange(1, num_frequencies + 1)
    angles = 2 * jnp.pi * value * frequencies / 8
    sin_features = jnp.sin(angles)
    cos_features = jnp.cos(angles)
    return jnp.concatenate([sin_features, cos_features])

@functools.partial(jax.vmap, in_axes=(0,), out_axes=0)
def process_vector(vector):
    """
    Processes the input vector by removing the first element,
    encoding it with Fourier features, and concatenating the result.

    Args:
        vector: A 1D JAX array.

    Returns:
        A 1D JAX array with the Fourier features concatenated to the remaining elements.
    """
    # Remove the first element
    first_element = vector[0]
    remaining_vector = vector[1:]

    # Encode the first element with Fourier features
    fourier_encoded = encode_fourier_features(first_element)

    # Concatenate the Fourier features with the remaining vector
    result_vector = jnp.concatenate([fourier_encoded, remaining_vector])

    return result_vector

class ActorCritic(nn.Module):
    """Model."""

    num_outputs: int

    @nn.compact
    def __call__(self, input):
        dtype = jnp.float32
        x = input['image'].astype(dtype) / 255.0
        x = einops.rearrange(x, 'bs stack height width channels -> bs height width (stack channels)')
        p_x = input["proprio"]
        p_x = process_vector(p_x)
        x = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            name='conv1',
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            name='conv2',
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            name='conv3',
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = jnp.concatenate([p_x, x], -1)
        x = nn.Dense(features=512, name='hidden', dtype=dtype)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.num_outputs, name='logits', dtype=dtype)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name='value', dtype=dtype)(x)
        return policy_log_probabilities, value


class TheFramestackPolicy(BaseAgent):
    def __init__(
            self,
            train_state: train_state.TrainState,
            framestack_len: int = 4,
            framestack_keys = ['image'],
            name='agent_0'):
        self.train_state = train_state
        self.key = jax.random.PRNGKey(0)
        self.history = collections.deque(maxlen=framestack_len)
        self.framestack_len =framestack_len
        self.framestack_keys = framestack_keys
        self.name = name

    def load(self, ckpt_dir: str) -> None:
        pass

    def get_action(self, observation: Dict[str, np.ndarray]) -> int:

        self.history.append(copy.copy(observation))
        while len(self.history) < self.framestack_len:
            self.history.append(copy.copy(observation))

        observation = copy.copy(observation)

        collated = smart_collate(self.history)
        for key in self.framestack_keys:
            observation[key] = collated[key]

        action, _, _ = policy_step(self.train_state, {self.name: observation}, self.key, deterministic=False)
        self.key, _ = jax.random.split(self.key)
        return action[self.name]

    def reset_state(self) -> None:
        self.history.clear()



@functools.partial(jax.jit, static_argnums=0)
def _policy_action(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    state: np.ndarray,
    proprio: np.ndarray,
):
    out = apply_fn({'params': params}, {'image': state, 'proprio': proprio})
    return out

def policy_action(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    observation,
):
    out = _policy_action(apply_fn, params, observation['image'], observation['proprio'])
    return out



def policy_step(
        train_state: train_state.TrainState,
        observation: np.ndarray,
        rng_key: jax.random.PRNGKey,
        deterministic: bool = False):
    """
    given an observation per agent makes a model's forward pass
    """

    # Apply the model to get the policy log probabilities and value estimate.

    keys = []
    values = []

    for k, v in observation.items():
        keys.append(k)
        values.append(v)

    obs_batched = smart_collate(values)

    # model_output = train_state.apply_fn({'params': train_state.params}, obs_batched)
    model_output = policy_action(train_state.apply_fn, train_state.params, obs_batched)
    log_probabilities, values = model_output
    if deterministic:
        actions = jnp.argmax(log_probabilities, -1)
    else:
        actions = jax.random.categorical(rng_key, log_probabilities)

    agent_acts = {}
    agent_logprobs = {}
    agent_value = {}

    for agent, a, lp, v in zip(keys, actions, log_probabilities, values):
        agent_acts[agent] = a.item()
        agent_logprobs[agent] = lp[a]
        agent_value[agent] = v


    return agent_acts, agent_logprobs, agent_value



def get_experience(train_state: train_state.TrainState, env, num_steps: int, rng_key: jax.random.PRNGKey) -> List[List[ExpTuple]]:
    """Collects a list of experience tuples for each parallel environment.

    Args:
        train_state: The flax TrainState object containing the model and parameters.
        env: VectorParallelEnv object, the vectorized environment.
        num_steps: Number of steps to run in each environment.
        rng_key: JAX random key for action sampling.

    Returns:
        A list of lists of ExpTuple, where each sublist represents one environment's trajectory.
    """
    # Reset environments and get initial states
    obs, _ = list(zip(*env.reset()))
    num_envs = len(obs)

    # Prepare all thread experiences
    all_experiences = [[] for _ in range(num_envs)]

    # We will split the RNG key for each environment's action sampling
    keys = jax.random.split(rng_key, num_envs)

    for step in tqdm(range(num_steps + 1), desc='collecting exp'):
        actions = []
        log_probs = []
        values = []

        # For every environment, generate the action, log_prob, and value
        for i, observation in enumerate(obs):
            if len(observation.keys()) == 0:
                continue
            action, log_prob, value = policy_step(train_state, observation, keys[i])
            actions.append(action)
            log_probs.append(log_prob)
            values.append({k: v[..., 0] for k, v in value.items()})

        # Take a step in all environments using the computed actions
        new_obs, rewards, terminations, truncs, infos = env.step(actions)

        dones = [{k:tm[k] or tr[k]  for k in tm.keys()} for tm, tr in zip(terminations, truncs)]
        for i in range(num_envs):
            exp_tuple = ExpTuple(
                state=obs[i],             # The current state
                action=actions[i],        # The action chosen by the policy
                reward=rewards[i],        # Reward from the environment
                value=values[i],          # Value estimate for the current state
                log_prob=log_probs[i],    # Log probability of the chosen action
                done=dones[i]             # Whether the episode has ended
            )
            # Save the experience tuple for the i-th environment
            all_experiences[i].append(exp_tuple)

        # Move onto the next observations (new state after step)
        obs = new_obs

        # Update random keys for the next step
        keys = jax.random.split(keys[0], num_envs)

    print("MEAN TRAIN NONZERO REWARDS", np.mean([v for tr in all_experiences for x in tr for v in x.reward.values() if v != 0]))
    print("MEAN TRAIN NONZERO LOGITS", np.mean([v for tr in all_experiences for x in tr for v in x.log_prob.values() if v != 0]))
    return all_experiences


def process_experience(
        experience: tp.List[tp.List[ExpTuple]],
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ):
    num_envs = len(experience)
    possible_agent_names = [f"agent_{i}" for i in range(8)]

    trajectories = []

    for agent_name in possible_agent_names:
        for env_i in range(num_envs):
            env_experience = experience[env_i]
            agent_obs = smart_collate([x.state[agent_name] for x in env_experience[:-1]])
            agent_act = smart_collate([x.action[agent_name] for x in env_experience[:-1]])
            agent_reward = smart_collate([x.reward[agent_name] for x in env_experience[:-1]])
            agent_value = smart_collate([x.value[agent_name] for x in env_experience])
            agent_logprob = smart_collate([x.log_prob[agent_name] for x in env_experience[:-1]])
            agent_done = ~smart_collate([x.done[agent_name] for x in env_experience[:-1]])
            agent_advantages = gae_advantages(agent_reward, agent_done, agent_value, gamma, lambda_)
            agent_returns = agent_advantages + agent_value[:-1]
            trajectories.append((agent_obs, agent_act, agent_logprob, agent_returns, agent_advantages))

    trajectories = tuple(smart_concat(x) for x in zip(*trajectories))
    return trajectories



