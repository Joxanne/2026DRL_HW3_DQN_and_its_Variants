"""Double DQN training (HW3-2) — Hasselt et al. 2016.

Decouples action selection (online net) from value estimation (target net)
to mitigate the systematic Q-value over-estimation of vanilla DQN.

    Y = r + gamma * (1 - done) * Q_target(s', argmax_a' Q_online(s', a'))

Target network is hard-synced from online every `sync_freq` training steps
(global gradient updates, not epochs).
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.model import build_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    save_metrics, set_seed,
)

STAGE_LABEL = 'HW3-2: Enhanced DQN Variants for player mode'


def train_double(
    *,
    epochs: int = 3000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,
    mode: str = 'player',
    seed: int = 42,
    snapshot_every: int = 150,
    out_dir: str = 'results/HW3-2/double_player',
    eval_n_games: int = 1000,
) -> dict:
    """Train Double DQN. Saves checkpoint, snapshots/, losses.npy, loss.png,
    and metrics.json under `out_dir`. Returns metrics dict.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    online_model = build_model()
    target_model = build_model()
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(online_model.parameters(), lr=lr)

    replay: deque = deque(maxlen=mem_size)
    losses: list[float] = []
    global_step = 0
    t0 = time.time()

    torch.save(online_model.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for i in tqdm(range(epochs), desc=f'double/{mode}'):
        game = Gridworld(size=4, mode=mode)
        state1 = encode_state(game)
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = online_model(state1)
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

                Q1 = online_model(state1_batch)
                with torch.no_grad():
                    online_next = online_model(state2_batch)
                    next_actions = online_next.argmax(dim=1, keepdim=True)
                    target_next = target_model(state2_batch)
                    next_q = target_next.gather(1, next_actions).squeeze(1)
                Y = reward_batch + gamma * (1 - done_batch) * next_q
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)
                ).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

                global_step += 1
                if global_step % sync_freq == 0:
                    target_model.load_state_dict(online_model.state_dict())

            if reward != -1 or mov > max_moves:
                status = 0

        if (i + 1) % snapshot_every == 0:
            torch.save(online_model.state_dict(),
                       snapshots_dir / f'epoch_{i + 1:04d}.pth')

    wall_time = time.time() - t0
    torch.save(online_model.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Double DQN ({mode} mode) — training loss')

    eval_result = evaluate(online_model, mode=mode, n_games=eval_n_games)
    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'double_{mode}',
        'mode': mode,
        'method': 'double',
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'epsilon': epsilon, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'sync_freq': sync_freq, 'seed': seed,
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
    parser = argparse.ArgumentParser(description='Double DQN training (HW3-2).')
    parser.add_argument('--mode', default='player',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--sync-freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=150)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-2/double_{args.mode}'
    train_double(
        epochs=args.epochs, gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, sync_freq=args.sync_freq,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
