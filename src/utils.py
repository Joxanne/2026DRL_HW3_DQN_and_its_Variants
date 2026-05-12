"""Shared utilities for HW3 DQN training and evaluation."""

import json
import random
from typing import Any

import numpy as np
import torch

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def set_seed(seed: int) -> None:
    """Seed Python `random`, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_state(game: Any) -> torch.Tensor:
    """Encode Gridworld state to (1, 64) float tensor with small uniform noise.
    Mirrors the book: render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.
    """
    arr = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    return torch.from_numpy(arr).float()


def epsilon_greedy(qval: torch.Tensor, epsilon: float, n_actions: int = 4) -> int:
    """Return action index. With prob `epsilon`, sample uniformly; else argmax."""
    if random.random() < epsilon:
        return int(np.random.randint(0, n_actions))
    return int(torch.argmax(qval).item())


def running_mean(x: np.ndarray, N: int = 50) -> np.ndarray:
    """Simple moving average of length N. Returns array of length len(x) - N.

    Matches DRL in Action Listing 3.6 verbatim; plot code is aligned to this
    length convention.
    """
    if len(x) <= N:
        return np.array([])
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i + N] @ conv) / N
    return y


def save_metrics(path: str, **kwargs: Any) -> None:
    """Dump kwargs as a JSON object to `path` (UTF-8, indent=2)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(kwargs, f, indent=2, ensure_ascii=False)


def test_model(model, mode: str = 'static', max_steps: int = 15) -> tuple[bool, int]:
    """Run one greedy (no exploration) test game in `mode`. Returns (won, steps).
    `won` is True iff the agent reached Goal within max_steps.
    """
    from src.gridworld_env import Gridworld

    game = Gridworld(size=4, mode=mode)
    state = encode_state(game)
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            qval = model(state)
        action_idx = int(torch.argmax(qval).item())
        action = ACTION_SET[action_idx]
        game.makeMove(action)
        state = encode_state(game)
        reward = game.reward()
        if reward != -1:
            return (reward > 0, step)
    return (False, max_steps)


test_model.__test__ = False  # type: ignore[attr-defined]


def evaluate(model, mode: str = 'static', n_games: int = 1000) -> dict:
    """Run `n_games` greedy test games. Returns win_rate, avg_steps_per_win."""
    n_wins = 0
    win_steps = []
    for _ in range(n_games):
        won, steps = test_model(model, mode=mode)
        if won:
            n_wins += 1
            win_steps.append(steps)
    win_rate = n_wins / n_games
    avg_steps = (sum(win_steps) / len(win_steps)) if win_steps else 0.0
    return {
        'win_rate': float(win_rate),
        'avg_steps_per_win': float(avg_steps),
        'n_games': int(n_games),
        'n_wins': int(n_wins),
    }
