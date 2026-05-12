"""DQN + Experience Replay Buffer training — Listing 3.5 of DRL in Action (HW3-1).

Introduces a replay buffer (deque) to break temporal correlation by sampling
random minibatches of past transitions for each gradient update.

Key improvement over naive DQN:
    - Transitions stored in buffer of size `mem_size`
    - Random minibatch of size `batch_size` sampled each step
    - Bellman target uses a second Q-network read (same model, no target net yet)
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dqn_naive import _plot_loss, STAGE_LABEL
from src.gridworld_env import Gridworld
from src.model import build_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    save_metrics, set_seed,
)


def train_replay(
    *,
    epochs: int = 5000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    mode: str = 'random',
    seed: int = 42,
    snapshot_every: int = 250,
    out_dir: str = 'results/HW3-1/replay_random',
    eval_n_games: int = 1000,
) -> dict:
    """Train DQN with experience replay per Listing 3.5. Saves checkpoint,
    snapshots, losses, loss.png, and metrics.json under `out_dir`.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    replay: deque = deque(maxlen=mem_size)
    losses: list[float] = []
    t0 = time.time()

    torch.save(model.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for i in tqdm(range(epochs), desc=f'replay/{mode}'):
        game = Gridworld(size=4, mode=mode)
        state1 = encode_state(game)
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = model(state1)
            action_idx = epsilon_greedy(qval, epsilon)
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            state2 = encode_state(game)
            reward = game.reward()
            done = reward > 0
            replay.append((state1, action_idx, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(list(replay), batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.tensor(
                    [r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.tensor(
                    [d for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)

                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                Y = reward_batch + gamma * (
                    (1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)
                ).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            if reward != -1 or mov > max_moves:
                status = 0

        if (i + 1) % snapshot_every == 0:
            torch.save(model.state_dict(),
                       snapshots_dir / f'epoch_{i + 1:04d}.pth')

    wall_time = time.time() - t0
    torch.save(model.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'DQN + Replay ({mode} mode) — training loss')

    eval_result = evaluate(model, mode=mode, n_games=eval_n_games)
    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'replay_{mode}',
        'mode': mode,
        'method': 'replay',
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'epsilon': epsilon, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'seed': seed,
            'snapshot_every': snapshot_every,
        },
        'final_loss_mean_last_100': float(tail.mean()) if len(tail) else 0.0,
        'final_loss_std_last_100': float(tail.std()) if len(tail) else 0.0,
        'win_rate': eval_result['win_rate'],
        'avg_steps_per_win': eval_result['avg_steps_per_win'],
        'n_eval_games': eval_result['n_games'],
        'training_wall_time_sec': float(wall_time),
    }
    save_metrics(str(out_path / 'metrics.json'), **metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='DQN + Replay training (HW3-1).')
    parser.add_argument('--mode', default='random',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=250)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-1/replay_{args.mode}'
    train_replay(
        epochs=args.epochs, gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
