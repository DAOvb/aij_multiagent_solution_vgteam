import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import functools 
import distrax

class ConvEncoder(nn.Module):
  """Class defining the convolution model."""

  @nn.compact
  def __call__(self, x):
    dtype = jnp.float32
    x = x.astype(dtype) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = x.reshape(x.shape[0],x.shape[1], -1) #flatten
    return x

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        image, vector = obs
        processed_image = ConvEncoder()(image)
        processed_vector = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(vector)
        full_obs = jax.lax.concatenate([processed_image, processed_vector], 2)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(full_obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi

class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        image, vector = world_state
        processed_image = ConvEncoder()(image)
        processed_vector = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(vector)
        full_obs = jax.lax.concatenate([processed_image, processed_vector], 2)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(full_obs)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config['NUM_CRITIC_OUTS'], kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)