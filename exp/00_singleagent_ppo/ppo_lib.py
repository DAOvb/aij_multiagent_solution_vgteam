
import functools
from collections.abc import Callable
from typing import Any, Dict

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from agent import policy_action
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import checkpoints, train_state
from tqdm import tqdm
from agent import get_experience, process_experience
from env import make_training_env


def loss_fn(
    params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    minibatch: tuple,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
    """Evaluate the loss function.

    Compute loss as a sum of three components: the negative of the PPO clipped
    surrogate objective, the value function loss and the negative of the entropy
    bonus.

    Args:
        params: the parameters of the actor-critic model
        apply_fn: the actor-critic model's apply function
        minibatch: tuple of five elements forming one experience batch:
                states: shape (batch_size, 84, 84, 4)
                actions: shape (batch_size, 84, 84, 4)
                old_log_probs: shape (batch_size,)
                returns: shape (batch_size,)
                advantages: shape (batch_size,)
        clip_param: the PPO clipping parameter used to clamp ratios in loss function
        vf_coeff: weighs value function loss in total loss
        entropy_coeff: weighs entropy bonus in the total loss

    Returns:
        loss: the PPO loss, scalar quantity
    """
    states, actions, old_log_probs, returns, advantages = minibatch

    log_probs, values = policy_action(apply_fn, params, states)
    values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
    probs = jnp.exp(log_probs)

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)
    entropy = jnp.sum(-probs * log_probs, axis=1).mean()

    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)

    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(
        1.0 - clip_param, ratios, 1.0 + clip_param
    )
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)


    return ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: train_state.TrainState,
    trajectories: tuple,
    batch_size: int,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
    """Compilable train step.

    Runs an entire epoch of training (i.e. the loop over minibatches within
    an epoch is included here for performance reasons).

    Args:
        state: the train state
        trajectories: tuple of the following five elements forming the experience:
                    states: shape (steps_per_agent*num_agents, 84, 84, 4)
                    actions: shape (steps_per_agent*num_agents, 84, 84, 4)
                    old_log_probs: shape (steps_per_agent*num_agents, )
                    returns: shape (steps_per_agent*num_agents, )
                    advantages: (steps_per_agent*num_agents, )
        batch_size: the minibatch size, static argument
        clip_param: the PPO clipping parameter used to clamp ratios in loss function
        vf_coeff: weighs value function loss in total loss
        entropy_coeff: weighs entropy bonus in the total loss

    Returns:
        optimizer: new optimizer after the parameters update
        loss: loss summed over training steps
    """
    iterations = trajectories[1].shape[0] // batch_size
    trajectories = jax.tree_util.tree_map(
        lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories
    )
    loss = 0.0
    for batch_i in tqdm(range(iterations)):
        batch = jax.tree_util.tree_map(
            lambda x: x[batch_i], trajectories
        )
        grad_fn = jax.value_and_grad(loss_fn)
        l, grads = grad_fn(
            state.params, state.apply_fn, batch, clip_param, vf_coeff, entropy_coeff
        )
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss


def create_train_state(
    params,
    model: nn.Module,
    config: ml_collections.ConfigDict,
    train_steps: int,
) -> train_state.TrainState:
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=train_steps,
        )
    else:
        lr = config.learning_rate
    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    return state



@functools.partial(jax.jit, static_argnums=1)
def get_initial_params(key: jax.Array, model: nn.Module):
  init_batch = {"image": jnp.ones((1, 4, 60, 60, 3), jnp.float32), "proprio": jnp.ones((1,7), jnp.float32)}
  initial_params = model.init(key, init_batch)['params']
  return initial_params



def train(
    model, config: ml_collections.ConfigDict, model_dir: str,
    eval_train_state_fn: Callable[[train_state.TrainState], Dict[str, float]]
):
    """Main training loop.

    Args:
        model: the actor-critic model
        config: object holding hyperparameters and the training information
        model_dir: path to dictionary where checkpoints and logging info are stored

    Returns:
        optimizer: the trained optimizer
    """

    simulators = make_training_env()
    summary_writer = tensorboard.SummaryWriter(model_dir)
    summary_writer.hparams(dict(config))
    loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
    log_frequency = config.eval_frequency
    checkpoint_frequency = config.checkpoint_frequency
    # train_step does multiple steps per call for better performance
    # compute number of steps per call here to convert between the number of
    # train steps and the inner number of optimizer steps
    iterations_per_step = (
        8 * config.num_agents * config.actor_steps // config.batch_size
    )

    initial_params = get_initial_params(jax.random.key(0), model)
    state = create_train_state(
        initial_params,
        model,
        config,
        loop_steps * config.num_epochs * iterations_per_step,
    )
    del initial_params
    state = checkpoints.restore_checkpoint(model_dir, state)
    # number of train iterations done by each train_step

    start_step = int(state.step) // config.num_epochs // iterations_per_step + 1
    logging.info('Start training from step: %s', start_step)

    key = jax.random.key(0)
    for step in range(start_step, loop_steps):
        # Bookkeeping and testing.
        if step % log_frequency == 0:
            eval_result = eval_train_state_fn(state, step)
            frames = step * config.num_agents * config.actor_steps
            for k, v in eval_result.items():
                summary_writer.scalar(k, v, frames)
            logging.info(
                'Step %s:\nframes seen %s\nscore %s\n\n', step, frames, eval_result['score']
            )

        # Core training code.
        alpha = (
            1.0 - step / loop_steps if config.decaying_lr_and_clip_param else 1.0
        )
        all_experiences = get_experience(state, simulators, config.actor_steps, key)
        key, _ = jax.random.split(key)
        trajectories = process_experience(
            all_experiences,
            config.gamma,
            config.lambda_,
        )
        clip_param = config.clip_param * alpha
        for _ in tqdm(range(config.num_epochs), desc="Train Epochs"):
            permutation =np.random.permutation(
                trajectories[1].shape[0]
            )
            trajectories = jax.tree_util.tree_map(
                lambda x: x[permutation], trajectories
            )
            state, _ = train_step(
                state,
                trajectories,
                config.batch_size,
                clip_param=clip_param,
                vf_coeff=config.vf_coeff,
                entropy_coeff=config.entropy_coeff,
            )
        if (step + 1) % checkpoint_frequency == 0:
            print(f'saved checkpoint on step {step + 1}!')
            checkpoints.save_checkpoint(model_dir, state, step + 1)

    return state

