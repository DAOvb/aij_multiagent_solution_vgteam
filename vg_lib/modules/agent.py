import flax.linen as nn
from flax.training.train_state import TrainState
from aij_multiagent_rl.agents import BaseAgent
import pickle
import jax.numpy as jnp
import jax

class FlaxAgent(BaseAgent):
    
    def __init__(self,):
        pass
    
    def load(self, ckpt_dir: str) -> None:
        from flax.core import freeze

        with open(ckpt_dir, 'rb') as handle:
            state_dict = pickle.load(handle)
        params = freeze(jax.tree_util.tree_map(lambda x: jnp.array(x), state_dict['params']))
        step = state_dict['step']
        apply_fn = 
        
        
        