from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import gymnasium as gym
import numpy as np


@dataclass
class ReplayStats:
    transitions_written: int = 0
    chunks_written: int = 0


class ReplayRecorder:
    """Persist self-generated transitions to compressed NPZ chunks.

    Nothing here is prior data. Every transition is created by the agent while
    it is interacting with the live game.
    """

    def __init__(self, directory: str | Path, chunk_size: int = 2000) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(100, int(chunk_size))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.stats = ReplayStats()
        self._counter = 0

        self._obs: list[np.ndarray] = []
        self._next_obs: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._episodes: list[int] = []
        self._steps: list[int] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        episode: int,
        episode_step: int,
    ) -> None:
        self._obs.append(np.asarray(obs, dtype=np.uint8).copy())
        self._next_obs.append(np.asarray(next_obs, dtype=np.uint8).copy())
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._episodes.append(int(episode))
        self._steps.append(int(episode_step))

        if len(self._actions) >= self.chunk_size:
            self.flush()

    def flush(self) -> Path | None:
        if not self._actions:
            return None

        self._counter += 1
        path = self.directory / f"replay_{self.session_id}_{self._counter:06d}.npz"

        np.savez_compressed(
            path,
            obs=np.stack(self._obs, axis=0),
            next_obs=np.stack(self._next_obs, axis=0),
            actions=np.asarray(self._actions, dtype=np.int8),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.bool_),
            episodes=np.asarray(self._episodes, dtype=np.int32),
            episode_steps=np.asarray(self._steps, dtype=np.int32),
        )

        count = len(self._actions)
        self.stats.transitions_written += count
        self.stats.chunks_written += 1

        self._obs.clear()
        self._next_obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._episodes.clear()
        self._steps.clear()

        return path


class PersistentReplayWrapper(gym.Wrapper):
    """Record live interaction while remaining transparent to the RL algorithm."""

    def __init__(
        self,
        env: gym.Env,
        directory: str | Path,
        chunk_size: int = 2000,
        enabled: bool = True,
    ) -> None:
        super().__init__(env)
        self.enabled = bool(enabled)
        self.recorder = ReplayRecorder(directory, chunk_size) if self.enabled else None
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.uint8)
        return obs, info

    def step(self, action: Any):
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before step().")

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        if self.recorder is not None:
            self.recorder.add(
                obs=self._last_obs,
                action=int(np.asarray(action).item()),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.uint8),
                done=done,
                episode=int(info.get("episode_index", -1)),
                episode_step=int(info.get("episode_steps", -1)),
            )

        self._last_obs = np.asarray(next_obs, dtype=np.uint8)
        return next_obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.recorder is not None:
            path = self.recorder.flush()
            if path is not None:
                print(f"Flushed replay data: {path}")
        self.env.close()
