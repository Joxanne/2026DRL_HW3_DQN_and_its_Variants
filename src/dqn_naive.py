"""Naive DQN training — Listing 3.3 of DRL in Action Ch.3 (HW3-1).

Updates the network on every single transition without any replay buffer.
Suffers from temporal correlation and catastrophic forgetting, but serves
as the baseline to compare against DQN+Replay.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.gridworld_env import Gridworld
from src.model import build_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    running_mean, save_metrics, set_seed,
)

STAGE_LABEL = 'HW3-1: Naive DQN for static mode'


def train_naive(
    *,
    epochs: int = 1000,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    lr: float = 1e-3,
    mode: str = 'static',
    seed: int = 42,
    snapshot_every: int = 50,
    out_dir: str = 'results/HW3-1/naive_static',
    eval_n_games: int = 1000,
) -> dict:
    """Train naive DQN per Listing 3.3. Saves checkpoint, snapshots, losses,
    loss.png, and metrics.json under `out_dir`. Returns metrics dict."""
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epsilon = epsilon_start
    losses: list[float] = []
    t0 = time.time()

    torch.save(model.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for i in tqdm(range(epochs), desc=f'naive/{mode}'):
        game = Gridworld(size=4, mode=mode)
        state = encode_state(game)
        status = 1
        while status == 1:
            qval = model(state)
            action_idx = epsilon_greedy(qval, epsilon)
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            state2 = encode_state(game)
            reward = game.reward()
            with torch.no_grad():
                newQ = model(state2)
            maxQ = torch.max(newQ)
            if reward == -1:
                Y = reward + (gamma * maxQ)
            else:
                Y = float(reward)
            Y = torch.tensor([Y]).detach()
            X = qval.squeeze()[action_idx]
            loss = loss_fn(X, Y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            state = state2
            if reward != -1:
                status = 0

        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epochs

        if (i + 1) % snapshot_every == 0:
            torch.save(model.state_dict(),
                       snapshots_dir / f'epoch_{i + 1:04d}.pth')

    wall_time = time.time() - t0

    torch.save(model.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)

    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Naive DQN ({mode} mode) — training loss')

    eval_result = evaluate(model, mode=mode, n_games=eval_n_games)

    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'naive_{mode}',
        'mode': mode,
        'method': 'naive',
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma,
            'epsilon_start': epsilon_start, 'epsilon_end': epsilon_end,
            'lr': lr, 'seed': seed, 'snapshot_every': snapshot_every,
        },
        'final_loss_mean_last_100': float(tail.mean()),
        'final_loss_std_last_100': float(tail.std()),
        'win_rate': eval_result['win_rate'],
        'avg_steps_per_win': eval_result['avg_steps_per_win'],
        'n_eval_games': eval_result['n_games'],
        'training_wall_time_sec': float(wall_time),
    }
    save_metrics(str(out_path / 'metrics.json'), **metrics)
    return metrics


def _plot_loss(losses: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, color='lightgray', linewidth=0.5, label='per-update loss')
    if len(losses) >= 50:
        sm = running_mean(losses, N=50)
        ax.plot(np.arange(50, 50 + len(sm)), sm, color='C0',
                linewidth=1.5, label='running mean (N=50)')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Naive DQN training (HW3-1).')
    parser.add_argument('--mode', default='static',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=50)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-1/naive_{args.mode}'
    train_naive(
        epochs=args.epochs, gamma=args.gamma, lr=args.lr,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
