from torch import nn, Tensor
import torch
from typing import Any, Sequence
import numpy as np


class CNNencoder(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

    def forward(
        self,
        x: np.ndarray | Tensor,
        rnn_state: Any | None = None,
    ) -> tuple[Tensor, Any]:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x.reshape(-1, self.c, self.w, self.h)), rnn_state


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        layers = [
            nn.init.orthogonal_(nn.Linear(input_dim, hidden_sizes[0])),
            nn.SiLU(inplace=True),
        ]
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    nn.init.orthogonal_(
                        nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
                    ),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.append(
            nn.init.orthogonal_(nn.Linear(hidden_sizes[-1], out_features=output_dim))
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: np.ndarray | Tensor) -> Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.layers(x)
