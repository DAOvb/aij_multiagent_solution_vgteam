from .common import MLP, CNNencoder
from torch import nn, Tensor
from typing import Sequence
import torch
import gymnasium as gym
import numpy as np
from aij_multiagent_rl.agents import BaseAgent
from vg_lib.utils.training import NUM_AGENTS


class Actor(nn.Module, BaseAgent):

    def __init__(
        self,
        obs_space_shape: Sequence[int],
        input_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        rnn_size: int,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.conv_encoder = CNNencoder(*obs_space_shape, device=device)
        self.proprio_encoder = nn.init.orthogonal_(
            nn.Linear(input_dim, hidden_sizes[0])
        )
        self.gru_input_dim = self.conv_encoder.output_dim + hidden_sizes[0]
        self.gru = nn.GRU(self.gru_input_dim, rnn_size, batch_first=True)
        rnn_state = torch.zeros(1, rnn_size, dtype=torch.float32, device=self.device)
        self.act_layers = MLP(
            input_dim=rnn_size,
            hidden_sizes=hidden_sizes[1:],
            device=device,
            output_dim=n_actions,
        )
        self.register_buffer(
            "initial_state", nn.Parameter(rnn_state, requires_grad=True)
        )
        self.register_buffer("current_state", rnn_state)

    def reset(self):
        self.current_state = self.initial_state

    def get_dist(self, logits: Tensor) -> torch.distributions.Distribution:
        dist = torch.distributions.Categorical(logits=logits)
        return dist

    def act(self, obs: dict[str, Tensor | np.ndarray], dones: np.ndarray[bool]):
        image = obs["image"]
        proprio = obs["proprio"]
        im_x = self.conv_encoder(image / 255.0)
        p_x = self.proprio_encoder(proprio)
        rnn_state = self.current_state * torch.as_tensor(
            1.0 - dones, dtype=torch.float32, device=self.device
        )
        rnn_state += self.initial_state * torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )
        _, rnn_state = self.gru(torch.cat([im_x, p_x], -1), rnn_state)
        self.current_state = rnn_state
        logits = self.act_layers(rnn_state)
        dist = self.get_dist(logits=logits)
        act = dist.sample()
        return act, dist.log_prob(act)


class CentralCritic(nn.Module):

    def __init__(
        self,
        obs_space_shape: Sequence[int],
        input_dim: int,
        hidden_sizes: Sequence[int],
        rnn_size: int,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.conv_encoder = CNNencoder(*obs_space_shape, device=device)
        self.proprio_encoder = nn.init.orthogonal_(
            nn.Linear(input_dim, hidden_sizes[0])
        )
        self.gru_input_dim = self.conv_encoder.output_dim + hidden_sizes[0]
        self.gru = nn.GRU(self.gru_input_dim, rnn_size, batch_first=True)
        rnn_state = torch.zeros(1, rnn_size, dtype=torch.float32, device=self.device)
        self.value = MLP(
            input_dim=rnn_size,
            hidden_sizes=hidden_sizes[1:],
            device=device,
            output_dim=NUM_AGENTS,
        )
        self.register_buffer(
            "initial_state", nn.Parameter(rnn_state, requires_grad=True)
        )
        self.register_buffer("current_state", rnn_state)

    def reset(self):
        self.current_state = self.initial_state

    def forward(self, obs: dict[str, Tensor | np.ndarray], dones: np.ndarray[bool]):
        image = obs["image"]
        proprio = obs["proprio"]
        im_x = self.conv_encoder(image / 255.0)
        p_x = self.proprio_encoder(proprio)
        rnn_state = self.current_state * torch.as_tensor(
            1.0 - dones, dtype=torch.float32, device=self.device
        )
        rnn_state += self.initial_state * torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )
        _, rnn_state = self.gru(torch.cat([im_x, p_x], -1), rnn_state)
        self.current_state = rnn_state
        return self.value(rnn_state)
