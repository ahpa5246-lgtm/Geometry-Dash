from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn


class PixelQNetwork(nn.Module):
    """Two-frame visual Q-network for hold/release actions."""

    def __init__(self, num_actions: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, 84, 84)
            encoded = int(self.encoder(dummy).shape[1])

        self.head = nn.Sequential(
            nn.Linear(encoded, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class DiskReplaySampler:
    """Random mini-batches from compressed replay chunks.

    A state is represented by two consecutive grayscale frames:
        [previous_frame, current_frame]
    so the offline learner can infer motion without game memory.
    """

    def __init__(self, replay_dir: str | Path, cache_files: int = 3) -> None:
        self.replay_dir = Path(replay_dir)
        self.files = sorted(self.replay_dir.glob("replay_*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No replay files found in {self.replay_dir}. "
                "Run live training first so the agent can generate its own data."
            )
        self.cache_files = max(1, int(cache_files))
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path in self._cache:
            item = self._cache.pop(path)
            self._cache[path] = item
            return item

        with np.load(path, allow_pickle=False) as z:
            item = {name: z[name] for name in z.files}

        self._cache[path] = item
        while len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)
        return item

    @staticmethod
    def _gray(frame_batch: np.ndarray) -> np.ndarray:
        if frame_batch.ndim != 4:
            raise ValueError(f"Expected observations [N,H,W,C], got {frame_batch.shape}")
        if frame_batch.shape[-1] == 1:
            return frame_batch[..., 0]
        if frame_batch.shape[-1] == 3:
            # RGB/BGR distinction is irrelevant for grayscale conversion here.
            return (
                0.299 * frame_batch[..., 0]
                + 0.587 * frame_batch[..., 1]
                + 0.114 * frame_batch[..., 2]
            ).astype(np.uint8)
        raise ValueError(f"Unsupported channel count: {frame_batch.shape[-1]}")

    def sample(self, batch_size: int, device: torch.device):
        batch_size = int(batch_size)

        for _ in range(20):
            path = random.choice(self.files)
            data = self._load(path)
            n = int(data["actions"].shape[0])
            if n < 2:
                continue

            episodes = data["episodes"]
            valid = np.flatnonzero(episodes[1:] == episodes[:-1]) + 1
            if valid.size == 0:
                continue

            idx = np.random.choice(valid, size=batch_size, replace=valid.size < batch_size)
            prev_idx = idx - 1

            obs_gray = self._gray(data["obs"])
            next_gray = self._gray(data["next_obs"])

            states = np.stack([obs_gray[prev_idx], obs_gray[idx]], axis=1)
            next_states = np.stack([obs_gray[idx], next_gray[idx]], axis=1)

            states_t = torch.from_numpy(states).to(device=device, dtype=torch.float32).div_(255.0)
            next_states_t = torch.from_numpy(next_states).to(device=device, dtype=torch.float32).div_(255.0)
            actions_t = torch.from_numpy(data["actions"][idx].astype(np.int64)).to(device)
            rewards_t = torch.from_numpy(data["rewards"][idx].astype(np.float32)).to(device)
            dones_t = torch.from_numpy(data["dones"][idx].astype(np.float32)).to(device)
            return states_t, actions_t, rewards_t, next_states_t, dones_t

        raise RuntimeError("Replay files do not contain enough consecutive transitions yet.")

    def transition_count(self) -> int:
        total = 0
        for path in self.files:
            data = self._load(path)
            total += int(data["actions"].shape[0])
        return total
