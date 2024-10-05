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
from agent import policy_action, critic_calc
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import checkpoints, train_state
from tqdm import tqdm
from agent import get_experience, process_experience
from env import make_training_env

def critic_loss_fn(
    critic_params: flax.core.FrozenDict,
    critic_apply_fn: Callable[..., Any],
    minibatch: tuple,
    clip_param: float,
    vf_coeff: float,
):
    """Evaluate the loss function.

    Compute loss as a sum of three components: the negative of the PPO clipped
    surrogate objective, the value function loss and the negative of the entropy
    bonus.
    """
    world_state, returns, batch_values = minibatch
    value = critic_calc(critic_apply_fn, critic_params, world_state)
    v_clip = batch_values + jax.lax.clamp(-clip_param, (value - batch_values), clip_param)
    vf1 = jnp.square(returns - value)
    vf2 = jnp.square(returns - v_clip)
    vf_loss = jnp.maximum(vf1, vf2)
    return vf_coeff * jnp.mean(vf_loss)

def policy_loss_fn(
    actor_params: flax.core.FrozenDict,
    actor_apply_fn: Callable[..., Any],
    minibatch: tuple,
    clip_param: float,
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
    states, actions, old_log_probs, advantages = minibatch
    
    log_probs = policy_action(actor_apply_fn, actor_params, states)
    probs = jnp.exp(log_probs)

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

    return ppo_loss - entropy_coeff * entropy


@functools.partial(jax.jit, static_argnums=(4,))
def train_step(
    state: train_state.TrainState,
    critic_state: train_state.TrainState,
    trajectories: tuple,
    world_traj: tuple,
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
    world_traj = jax.tree_util.tree_map(
        lambda x: x.reshape((iterations, batch_size // 8) + x.shape[1:]), world_traj
    )
    policy_loss, value_loss = 0.0, 0.0
    for batch_i in tqdm(range(iterations)):
        batch = jax.tree_util.tree_map(lambda x: x[batch_i], trajectories)
        world_batch = jax.tree_util.tree_map(lambda x: x[batch_i], world_traj)
        #update actor
        policy_grad_fn = jax.value_and_grad(policy_loss_fn)
        pl, pgrads = policy_grad_fn(
            state.params, state.apply_fn, batch, clip_param, entropy_coeff
        )
        policy_loss += pl
        state = state.apply_gradients(grads=pgrads)
        #update critic
        value_grad_fn = jax.value_and_grad(critic_loss_fn)
        vl, vgrads = value_grad_fn(
            critic_state.params, critic_state.apply_fn, world_batch, clip_param, vf_coeff
        )
        value_loss += vl
        critic_state = critic_state.apply_gradients(grads=vgrads)
        
        
    return state, critic_state, policy_loss, value_loss


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
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


@functools.partial(jax.jit, static_argnums=1)
def get_initial_actor_params(key: jax.Array, model: nn.Module):
    init_batch = {"image": jnp.ones((1, 4, 60, 60, 3), jnp.float32), "proprio": jnp.ones((1,4,7), jnp.float32)}
    initial_params = model.init(key, init_batch)["params"]
    return initial_params

@functools.partial(jax.jit, static_argnums=1)
def get_initial_critic_params(key: jax.Array, model: nn.Module):
    init_batch = {"image": jnp.ones((4, 110, 110, 3), jnp.float32), "wealth": jnp.zeros((4,8), jnp.float32), 
                  "has_resource": jnp.zeros((4,8), jnp.float32), "has_trash":jnp.zeros((4,8), jnp.float32)}
    initial_params = model.init(key, init_batch)["params"]
    return initial_params

def train(
    model,
    critic_model,
    config: ml_collections.ConfigDict,
    model_dir: str,
    eval_train_state_fn: Callable[[train_state.TrainState], Dict[str, float]],
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

    initial_params = get_initial_actor_params(jax.random.key(0), model)
    c_initial_params = get_initial_critic_params(jax.random.key(0), critic_model)
    state = create_train_state(
        initial_params,
        model,
        config,
        loop_steps * config.num_epochs * iterations_per_step,
    )
    critic_state = create_train_state(
        c_initial_params,
        critic_model,
        config,
        loop_steps * config.num_epochs * iterations_per_step,
    )
    del initial_params
    del c_initial_params
    state = checkpoints.restore_checkpoint(model_dir + "/a", state)
    critic_state = checkpoints.restore_checkpoint(model_dir + "/c", critic_state)
    # number of train iterations done by each train_step

    start_step = int(state.step) // config.num_epochs // iterations_per_step + 1
    logging.info("Start training from step: %s", start_step)

    key = jax.random.key(0)
    for step in range(start_step, loop_steps):
        # Bookkeeping and testing.
        if step % log_frequency == 0:
            eval_result = eval_train_state_fn(state, step)
            frames = step * config.num_agents * config.actor_steps
            for k, v in eval_result.items():
                summary_writer.scalar(k, v, frames)
            logging.info(
                "Step %s:\nframes seen %s\nscore %s\n\n",
                step,
                frames,
                eval_result["score"],
            )

        # Core training code.
        alpha = 1.0 - step / loop_steps if config.decaying_lr_and_clip_param else 1.0
        all_experiences = get_experience(
            state, critic_state, simulators, config.actor_steps, key
        )
        key, _ = jax.random.split(key)
        trajectories, world_trajectories = process_experience(
            all_experiences,
            config.gamma,
            config.lambda_,
        )

        clip_param = config.clip_param * alpha
        for _ in tqdm(range(config.num_epochs), desc="Train Epochs"):
            permutation = np.random.permutation(trajectories[1].shape[0])
            trajectories = jax.tree_util.tree_map(
                lambda x: x[permutation], trajectories
            )
            w_permutation = np.random.permutation(world_trajectories[1].shape[0])
            world_trajectories = jax.tree_util.tree_map(
                lambda x: x[w_permutation], world_trajectories
            )
            state, critic_state, policy_loss, value_loss = train_step(
                state=state,
                critic_state=critic_state,
                trajectories=trajectories,
                world_traj=world_trajectories,
                batch_size=config.batch_size,
                clip_param=clip_param,
                vf_coeff=config.vf_coeff,
                entropy_coeff=config.entropy_coeff,
            )
            summary_writer.scalar('pg_loss', policy_loss, step)
            summary_writer.scalar('critic_loss', value_loss, step)
        if (step + 1) % checkpoint_frequency == 0:
            print(f"saved checkpoint on step {step + 1}!")
            checkpoints.save_checkpoint(model_dir + "/a", state, step + 1)
            checkpoints.save_checkpoint(model_dir + "/c", state, step + 1)

    return state
