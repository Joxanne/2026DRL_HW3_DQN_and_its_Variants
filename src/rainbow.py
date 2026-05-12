"""Rainbow DQN — combines six improvements (HW3-4 bonus).

Components stacked in this implementation:
  1. Double DQN         — decoupled action selection / value estimation
  2. Dueling networks   — V(s) + A(s,a) heads
  3. Prioritized Replay — SumTree-based proportional PER with IS weights
  4. N-step returns     — bootstrapped over n=3 steps
  5. Distributional RL  — C51 (51-atom categorical distribution)
  6. Noisy Networks     — factorised Gaussian noise replaces epsilon-greedy

Reference: Hessel et al. 2017 "Rainbow: Combining Improvements in DRL"
"""

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.utils import ACTION_SET, encode_state, evaluate, save_metrics, set_seed

STAGE_LABEL = 'HW3-4: Rainbow DQN for Random Mode GridWorld'

# ---------------------------------------------------------------------------
# 1. Noisy Linear Layer (factorised Gaussian noise, Fortunato et al. 2018)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Linear layer with learnable per-weight Gaussian noise.

    Uses factorised noise: noise = f(eps_p) outer f(eps_q) where
    f(x) = sgn(x) * sqrt(|x|).  Only p+q random samples are drawn
    instead of p*q, reducing computation.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_eps', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_eps', torch.empty(out_features))

        self.sigma_init = sigma_init
        self._reset_parameters()
        self.sample_noise()

    def _reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_p = self._f(torch.randn(self.in_features))
        eps_q = self._f(torch.randn(self.out_features))
        self.weight_eps.copy_(eps_q.outer(eps_p))
        self.bias_eps.copy_(eps_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# 2. Distributional Dueling Network with Noisy Layers (C51)
# ---------------------------------------------------------------------------

class DistributionalDuelingMLP(nn.Module):
    """Combines C51 (distributional), Dueling, and Noisy networks.

    Output shape: (batch, n_actions, n_atoms) — softmax probabilities.
    Q-values for action selection: sum over atoms of z * p(s,a).
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden1: int = 150,
        hidden2: int = 100,
        n_actions: int = 4,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer(
            'support', torch.linspace(v_min, v_max, n_atoms))

        self.trunk = nn.Sequential(
            NoisyLinear(in_dim, hidden1), nn.ReLU(),
            NoisyLinear(hidden1, hidden2), nn.ReLU(),
        )
        self.value_head = NoisyLinear(hidden2, n_atoms)
        self.advantage_head = NoisyLinear(hidden2, n_actions * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h).view(-1, 1, self.n_atoms)          # (B,1,A)
        adv = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = v + (adv - adv.mean(dim=1, keepdim=True))        # (B,Na,A)
        return F.softmax(q_atoms, dim=2)                            # probabilities

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar Q(s,a) = sum_z z * p(s,a,z)."""
        probs = self.forward(x)                                     # (B,Na,A)
        return (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


def build_rainbow_model(**kwargs) -> DistributionalDuelingMLP:
    return DistributionalDuelingMLP(**kwargs)


# ---------------------------------------------------------------------------
# 3. SumTree (priority-based sampling in O(log N))
# ---------------------------------------------------------------------------

class SumTree:
    """Binary heap where each leaf stores a priority.  Internal nodes hold
    the sum of their subtree.  Supports O(log N) update and sample.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def _propagate(self, idx: int, delta: float):
        parent = idx >> 1
        while parent >= 1:
            self.tree[parent] += delta
            parent >>= 1

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float, data):
        leaf_idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get(self, s: float) -> tuple[int, float, object]:
        """Retrieve (leaf_index, priority, data) for cumulative value s."""
        idx = 1
        while idx < self.capacity:
            left = idx << 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, float(self.tree[idx]), self.data[data_idx]


# ---------------------------------------------------------------------------
# 4. Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Proportional prioritization replay buffer with IS weight correction.

    alpha: priority exponent (0=uniform, 1=fully prioritized)
    beta:  IS exponent, annealed from beta_start -> 1.0
    """

    def __init__(
        self,
        capacity: int = 1000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        self._max_priority = 1.0

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start +
                   self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, transition):
        self.tree.add(self._max_priority ** self.alpha, transition)

    def sample(self, batch_size: int) -> tuple:
        idxs, priorities, transitions = [], [], []
        segment = self.tree.total / batch_size
        self.frame += 1

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, t = self.tree.get(s)
            if t is None:
                continue
            idxs.append(idx)
            priorities.append(p)
            transitions.append(t)

        if not idxs:
            return None

        probs = np.array(priorities) / self.tree.total
        is_weights = (self.tree.n_entries * probs) ** (-self.beta)
        is_weights /= is_weights.max()

        states = torch.cat([t[0] for t in transitions])
        actions = torch.tensor([t[1] for t in transitions])
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
        next_states = torch.cat([t[3] for t in transitions])
        dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32)
        weights = torch.tensor(is_weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, weights, idxs

    def update_priorities(self, idxs: list[int], td_errors: np.ndarray):
        for idx, err in zip(idxs, td_errors):
            priority = (abs(err) + self.epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)


# ---------------------------------------------------------------------------
# 5. N-step Buffer
# ---------------------------------------------------------------------------

class NStepBuffer:
    """Accumulates n transitions and returns the n-step bootstrapped tuple."""

    def __init__(self, n: int = 3, gamma: float = 0.9):
        self.n = n
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n)

    def add(self, transition) -> tuple | None:
        """Add a transition; return n-step tuple when buffer is full."""
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None
        R = 0.0
        for i, (s, a, r, s2, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:
                break
        s0, a0 = self.buffer[0][0], self.buffer[0][1]
        sn, dn = self.buffer[-1][3], self.buffer[-1][4]
        return (s0, a0, R, sn, dn)

    def flush(self) -> list[tuple]:
        """Drain remaining transitions at episode end."""
        results = []
        while len(self.buffer) > 0:
            R = 0.0
            for i, (s, a, r, s2, d) in enumerate(self.buffer):
                R += (self.gamma ** i) * r
                if d:
                    break
            s0, a0 = self.buffer[0][0], self.buffer[0][1]
            sn, dn = self.buffer[-1][3], self.buffer[-1][4]
            results.append((s0, a0, R, sn, dn))
            self.buffer.popleft()
        return results


# ---------------------------------------------------------------------------
# 6. Categorical (distributional) Bellman projection
# ---------------------------------------------------------------------------

def categorical_projection(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_probs: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Project the distributional Bellman target onto the fixed support.

    Args:
        rewards:    (B,) float
        dones:      (B,) float {0,1}
        next_probs: (B, n_atoms) probability mass under next greedy action
        support:    (n_atoms,) fixed atom values
        gamma:      discount factor

    Returns:
        m:          (B, n_atoms) projected target distribution
    """
    batch_size = rewards.size(0)
    n_atoms = support.size(0)
    v_min, v_max = support[0].item(), support[-1].item()
    delta_z = (v_max - v_min) / (n_atoms - 1)

    Tz = rewards.unsqueeze(1) + gamma * (1 - dones).unsqueeze(1) * support.unsqueeze(0)
    Tz = Tz.clamp(v_min, v_max)

    b = (Tz - v_min) / delta_z
    l = b.floor().long().clamp(0, n_atoms - 1)
    u = b.ceil().long().clamp(0, n_atoms - 1)

    m = torch.zeros(batch_size, n_atoms, device=rewards.device)
    offset = torch.arange(batch_size, device=rewards.device).unsqueeze(1) * n_atoms

    m.view(-1).scatter_add_(
        0, (l + offset).view(-1),
        (next_probs * (u.float() - b)).view(-1))
    m.view(-1).scatter_add_(
        0, (u + offset).view(-1),
        (next_probs * (b - l.float())).view(-1))
    return m


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_rainbow(
    *,
    epochs: int = 5000,
    gamma: float = 0.9,
    lr: float = 5e-4,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,
    n_step: int = 3,
    n_atoms: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    alpha: float = 0.6,
    beta_start: float = 0.4,
    mode: str = 'random',
    seed: int = 42,
    snapshot_every: int = 250,
    out_dir: str = 'results/HW3-4/rainbow_random',
    eval_n_games: int = 1000,
) -> dict:
    """Train Rainbow DQN on Gridworld. Returns metrics dict."""
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    online = DistributionalDuelingMLP(
        n_atoms=n_atoms, v_min=v_min, v_max=v_max)
    target = DistributionalDuelingMLP(
        n_atoms=n_atoms, v_min=v_min, v_max=v_max)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    replay = PrioritizedReplayBuffer(
        capacity=mem_size, alpha=alpha, beta_start=beta_start,
        beta_frames=epochs * max_moves)
    n_step_buf = NStepBuffer(n=n_step, gamma=gamma)

    losses: list[float] = []
    global_step = 0
    t0 = time.time()

    torch.save(online.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for i in tqdm(range(epochs), desc=f'rainbow/{mode}'):
        game = Gridworld(size=4, mode=mode)
        state = encode_state(game)
        online.train()
        online.sample_noise()
        mov = 0

        while True:
            mov += 1
            with torch.no_grad():
                q = online.q_values(state)
            action_idx = int(q.argmax().item())
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            next_state = encode_state(game)
            reward = game.reward()
            done = reward > 0

            nstep = n_step_buf.add((state, action_idx, reward, next_state, done))
            if nstep is not None:
                replay.add(nstep)
            state = next_state

            if reward != -1 or mov >= max_moves:
                for t in n_step_buf.flush():
                    replay.add(t)
                break

            if replay.tree.n_entries >= batch_size:
                result = replay.sample(batch_size)
                if result is None:
                    continue
                s1, a, r, s2, d, is_w, idxs = result

                online.sample_noise()
                target.sample_noise()

                with torch.no_grad():
                    target.eval()
                    next_q = online.q_values(s2)
                    best_actions = next_q.argmax(dim=1)
                    target_probs_all = target(s2)
                    next_probs = target_probs_all[
                        torch.arange(batch_size), best_actions]

                m = categorical_projection(
                    r, d, next_probs, online.support, gamma ** n_step)

                online.train()
                log_probs_all = torch.log(online(s1) + 1e-8)
                log_probs = log_probs_all[torch.arange(batch_size), a]

                loss_per = -(m * log_probs).sum(dim=1)
                loss = (is_w * loss_per).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

                td_errors = loss_per.detach().cpu().numpy()
                replay.update_priorities(idxs, td_errors)

                global_step += 1
                if global_step % sync_freq == 0:
                    target.load_state_dict(online.state_dict())

        if (i + 1) % snapshot_every == 0:
            torch.save(online.state_dict(),
                       snapshots_dir / f'epoch_{i + 1:04d}.pth')

    wall_time = time.time() - t0
    torch.save(online.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Rainbow DQN ({mode} mode) — training loss')

    online.eval()

    def _rainbow_model_wrapper():
        class Wrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                return self.net.q_values(x)
        return Wrapper(online)

    wrapped = _rainbow_model_wrapper()
    eval_result = evaluate(wrapped, mode=mode, n_games=eval_n_games)

    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'rainbow_{mode}',
        'mode': mode,
        'method': 'rainbow',
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'sync_freq': sync_freq,
            'n_step': n_step, 'n_atoms': n_atoms,
            'v_min': v_min, 'v_max': v_max,
            'alpha': alpha, 'beta_start': beta_start,
            'seed': seed, 'snapshot_every': snapshot_every,
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
    parser = argparse.ArgumentParser(description='Rainbow DQN training (HW3-4).')
    parser.add_argument('--mode', default='random',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--sync-freq', type=int, default=500)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--n-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.0)
    parser.add_argument('--v-max', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta-start', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=250)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-4/rainbow_{args.mode}'
    train_rainbow(
        epochs=args.epochs, gamma=args.gamma, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, sync_freq=args.sync_freq,
        n_step=args.n_step, n_atoms=args.n_atoms,
        v_min=args.v_min, v_max=args.v_max,
        alpha=args.alpha, beta_start=args.beta_start,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
