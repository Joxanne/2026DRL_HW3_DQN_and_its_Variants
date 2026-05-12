"""Lightning-wrapped Combined DQN with optional training tricks (HW3-3).

Converts the Double+Dueling DQN from HW3-2 to PyTorch Lightning and
experiments with three optional training improvements:
  - Gradient norm clipping  (max_norm=10.0)
  - Cosine annealing LR schedule
  - Huber loss (SmoothL1Loss) instead of MSE

Five configurations are run on the random mode environment:
  baseline  — no tricks
  clip      — + gradient clipping
  sched     — + cosine annealing
  huber     — + Huber loss
  full      — all three tricks combined
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.model import build_dueling_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    save_metrics, set_seed,
)

STAGE_LABEL = 'HW3-3: Enhanced DQN for random mode with Training Tips'


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RolloutDataset(IterableDataset):
    """Plays one Gridworld game per call, fills a shared replay buffer, and
    yields individual (s, a, r, s', done) transitions from a random minibatch.
    DataLoader should have batch_size equal to the desired minibatch size.
    """

    def __init__(
        self,
        replay: deque,
        model_ref: nn.Module,
        epsilon: float,
        mode: str,
        max_moves: int,
        min_replay: int,
    ):
        self.replay = replay
        self.model_ref = model_ref
        self.epsilon = epsilon
        self.mode = mode
        self.max_moves = max_moves
        self.min_replay = min_replay

    def __iter__(self):
        game = Gridworld(size=4, mode=self.mode)
        state = encode_state(game)
        mov = 0
        while True:
            mov += 1
            with torch.no_grad():
                qval = self.model_ref(state)
            action_idx = epsilon_greedy(qval, self.epsilon)
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            next_state = encode_state(game)
            reward = game.reward()
            done = reward > 0
            self.replay.append((state, action_idx, reward, next_state, done))
            state = next_state
            if reward != -1 or mov >= self.max_moves:
                break

        if len(self.replay) >= self.min_replay:
            for item in self.replay:
                yield item


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class DQNLightningModule(pl.LightningModule):
    def __init__(
        self,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        lr: float = 1e-3,
        mem_size: int = 1000,
        batch_size: int = 200,
        max_moves: int = 50,
        sync_freq: int = 500,
        mode: str = 'random',
        use_huber: bool = False,
        use_cosine_sched: bool = False,
        total_epochs: int = 5000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.online_model = build_dueling_model()
        self.target_model = build_dueling_model()
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        self.loss_fn = nn.SmoothL1Loss() if use_huber else nn.MSELoss()
        self.replay: deque = deque(maxlen=mem_size)
        self._global_step_count = 0
        self._losses: list[float] = []

    def forward(self, x):
        return self.online_model(x)

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch

        states = states.squeeze(1)
        next_states = next_states.squeeze(1)
        rewards = rewards.float()
        dones = dones.float()

        Q1 = self.online_model(states)
        with torch.no_grad():
            online_next = self.online_model(next_states)
            next_actions = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target_model(next_states)
            next_q = target_next.gather(1, next_actions).squeeze(1)

        Y = rewards + self.hparams.gamma * (1 - dones) * next_q
        X = Q1.gather(1, actions.long().unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(X, Y.detach())
        self.log('train_loss', loss, prog_bar=True)
        self._losses.append(float(loss.item()))

        self._global_step_count += 1
        if self._global_step_count % self.hparams.sync_freq == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_model.parameters(), lr=self.hparams.lr)
        if self.hparams.use_cosine_sched:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.total_epochs)
            return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self):
        dataset = RolloutDataset(
            replay=self.replay,
            model_ref=self.online_model,
            epsilon=self.hparams.epsilon,
            mode=self.hparams.mode,
            max_moves=self.hparams.max_moves,
            min_replay=self.hparams.batch_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=_collate_transitions,
            num_workers=0,
        )


def _collate_transitions(batch):
    """Collate a list of (state, action, reward, next_state, done) tuples."""
    states = torch.cat([b[0] for b in batch])
    actions = torch.tensor([b[1] for b in batch])
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    next_states = torch.cat([b[3] for b in batch])
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return states, actions, rewards, next_states, dones


# ---------------------------------------------------------------------------
# Snapshot Callback
# ---------------------------------------------------------------------------

class SnapshotCallback(Callback):
    def __init__(self, snapshots_dir: Path, every: int):
        self.snapshots_dir = snapshots_dir
        self.every = every

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every == 0:
            path = self.snapshots_dir / f'epoch_{epoch:04d}.pth'
            torch.save(pl_module.online_model.state_dict(), path)


# ---------------------------------------------------------------------------
# train_lightning
# ---------------------------------------------------------------------------

def train_lightning(
    *,
    epochs: int = 5000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,
    mode: str = 'random',
    seed: int = 42,
    snapshot_every: int = 250,
    out_dir: str = 'results/HW3-3/baseline_random',
    eval_n_games: int = 1000,
    use_huber: bool = False,
    use_cosine_sched: bool = False,
    grad_clip: float | None = None,
    variant_name: str = 'baseline',
) -> dict:
    """Train one Lightning DQN variant. Returns metrics dict."""
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    module = DQNLightningModule(
        gamma=gamma, epsilon=epsilon, lr=lr,
        mem_size=mem_size, batch_size=batch_size,
        max_moves=max_moves, sync_freq=sync_freq, mode=mode,
        use_huber=use_huber, use_cosine_sched=use_cosine_sched,
        total_epochs=epochs,
    )

    torch.save(module.online_model.state_dict(),
               snapshots_dir / 'epoch_0000.pth')

    snapshot_cb = SnapshotCallback(snapshots_dir, every=snapshot_every)

    trainer_kwargs: dict = {
        'max_epochs': epochs,
        'callbacks': [snapshot_cb],
        'enable_progress_bar': True,
        'enable_model_summary': False,
        'enable_checkpointing': False,
        'logger': False,
    }
    if grad_clip is not None:
        trainer_kwargs['gradient_clip_val'] = grad_clip

    t0 = time.time()
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module)
    wall_time = time.time() - t0

    torch.save(module.online_model.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(module._losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Lightning DQN — {variant_name} ({mode} mode)')

    eval_result = evaluate(module.online_model, mode=mode, n_games=eval_n_games)
    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr

    tricks = []
    if grad_clip is not None:
        tricks.append(f'grad_clip={grad_clip}')
    if use_cosine_sched:
        tricks.append('cosine_sched')
    if use_huber:
        tricks.append('huber_loss')

    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'{variant_name}_{mode}',
        'mode': mode,
        'method': f'lightning_{variant_name}',
        'tricks': tricks,
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'epsilon': epsilon, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'sync_freq': sync_freq, 'seed': seed,
            'snapshot_every': snapshot_every,
            'use_huber': use_huber, 'use_cosine_sched': use_cosine_sched,
            'grad_clip': grad_clip,
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


# ---------------------------------------------------------------------------
# CLI — runs all five HW3-3 variants
# ---------------------------------------------------------------------------

_VARIANTS = {
    'baseline': dict(use_huber=False, use_cosine_sched=False, grad_clip=None),
    'clip':     dict(use_huber=False, use_cosine_sched=False, grad_clip=10.0),
    'sched':    dict(use_huber=False, use_cosine_sched=True,  grad_clip=None),
    'huber':    dict(use_huber=True,  use_cosine_sched=False, grad_clip=None),
    'full':     dict(use_huber=True,  use_cosine_sched=True,  grad_clip=10.0),
}


def main():
    parser = argparse.ArgumentParser(
        description='Lightning DQN with training tricks (HW3-3).')
    parser.add_argument('--variant', default='baseline',
                        choices=list(_VARIANTS.keys()),
                        help='Which trick configuration to run.')
    parser.add_argument('--mode', default='random',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--sync-freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=250)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    tricks = _VARIANTS[args.variant]
    out_dir = args.out_dir or f'results/HW3-3/{args.variant}_{args.mode}'
    train_lightning(
        epochs=args.epochs, gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, sync_freq=args.sync_freq,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
        variant_name=args.variant,
        **tricks,
    )


if __name__ == '__main__':
    main()
