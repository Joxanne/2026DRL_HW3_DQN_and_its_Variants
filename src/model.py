"""DQN MLP architecture (Listing 3.2 of DRL in Action Ch.3)."""

import torch
import torch.nn as nn


def build_model(in_dim: int = 64, hidden1: int = 150,
                hidden2: int = 100, out_dim: int = 4) -> nn.Sequential:
    """Two-hidden-layer MLP with ReLU. Matches Listing 3.2 defaults:
    64 (4-piece x 4x4 grid one-hot, flattened) -> 150 -> 100 -> 4 actions.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, out_dim),
    )


class DuelingMLP(nn.Module):
    """Dueling network: shared trunk -> V(s) head + A(s,a) head, combined with
    mean-baseline aggregation (Wang et al. 2016, eq. 9).

        Q(s, a) = V(s) + ( A(s, a) - mean_a A(s, a) )
    """

    def __init__(self, in_dim: int = 64, hidden1: int = 150,
                 hidden2: int = 100, n_actions: int = 4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden2, 1)
        self.advantage_head = nn.Linear(hidden2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)                              # (B, 1)
        a = self.advantage_head(h)                          # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))        # (B, n_actions)


def build_dueling_model(in_dim: int = 64, hidden1: int = 150,
                        hidden2: int = 100, n_actions: int = 4) -> DuelingMLP:
    """Dueling-network factory with the same trunk as `build_model` for fair
    comparison. Q = V + (A - mean A) per Wang et al. 2016, eq. 9.
    """
    return DuelingMLP(in_dim, hidden1, hidden2, n_actions)
