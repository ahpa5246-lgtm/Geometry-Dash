from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .input import GameController
from .vision import MotionEndDetector
from .window import ScreenCapture


class GeometryDashEnv(gym.Env[np.ndarray, int]):
    """Live, pixel-only Gymnasium environment for Geometry Dash.

    Observation:
        Current game-window pixels resized to the configured resolution.

    Actions:
        0 -> release jump
        1 -> hold jump

    Reward:
        Small positive reward for every step the run remains alive.
        A terminal penalty is applied when the screen becomes static.

    The agent never receives player coordinates, obstacle labels, level timing,
    or demonstrations.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        win_cfg = cfg["window"]
        input_cfg = cfg["input"]
        env_cfg = cfg["environment"]
        paths_cfg = cfg["paths"]

        self.capture = ScreenCapture(
            title_contains=str(win_cfg["title_contains"]),
            width=int(win_cfg["observation_width"]),
            height=int(win_cfg["observation_height"]),
            crop=win_cfg.get("crop"),
            grayscale=bool(win_cfg.get("grayscale", True)),
        )
        self.controller = GameController(
            action_key=str(input_cfg.get("key", "space")),
            start_key=str(input_cfg.get("start_key", "space")),
        )
        self.detector = MotionEndDetector(
            threshold=float(env_cfg["freeze_motion_threshold"]),
            freeze_frames=int(env_cfg["freeze_frames"]),
            warmup_frames=int(env_cfg["detector_warmup_frames"]),
        )

        channels = 1 if bool(win_cfg.get("grayscale", True)) else 3
        height = int(win_cfg["observation_height"])
        width = int(win_cfg["observation_width"])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, channels),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(2)

        self.step_hz = float(env_cfg["step_hz"])
        self.step_interval = 1.0 / self.step_hz
        self.alive_reward = float(env_cfg["alive_reward"])
        self.death_penalty = float(env_cfg["death_penalty"])
        self.max_episode_steps = max(
            1,
            int(float(env_cfg["max_episode_seconds"]) * self.step_hz),
        )
        self.reset_wait = float(env_cfg["reset_wait_seconds"])
        self.post_start_wait = float(env_cfg["post_start_wait_seconds"])

        self.episode_index = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.best_steps = 0
        self._last_step_clock = 0.0
        self._closed = False

        runs_dir = Path(paths_cfg["runs_dir"])
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.episode_csv = runs_dir / "episodes.csv"
        self._ensure_log_header()

    def _ensure_log_header(self) -> None:
        if self.episode_csv.exists():
            return
        with self.episode_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "steps",
                    "seconds",
                    "reward",
                    "best_seconds",
                    "termination_reason",
                    "final_motion",
                ]
            )

    def _append_episode(self, reason: str, final_motion: float) -> None:
        self.best_steps = max(self.best_steps, self.episode_steps)
        with self.episode_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.episode_index,
                    self.episode_steps,
                    round(self.episode_steps / self.step_hz, 3),
                    round(self.episode_reward, 6),
                    round(self.best_steps / self.step_hz, 3),
                    reason,
                    round(final_motion, 8),
                ]
            )

    def _wait_for_step_clock(self) -> None:
        now = time.perf_counter()
        due = self._last_step_clock + self.step_interval
        if due > now:
            time.sleep(due - now)
        self._last_step_clock = time.perf_counter()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("Environment is already closed.")

        self.controller.release()
        self.capture.focus()
        time.sleep(self.reset_wait)
        self.controller.restart()
        time.sleep(self.post_start_wait)

        self.detector.reset()
        self.episode_index += 1
        self.episode_steps = 0
        self.episode_reward = 0.0
        self._last_step_clock = time.perf_counter()

        obs = self.capture.grab_observation()
        self.detector.update(obs)
        return obs, {
            "episode_index": self.episode_index,
            "pixel_only": True,
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._wait_for_step_clock()
        self.controller.set_action(int(action))

        obs = self.capture.grab_observation()
        ended, motion = self.detector.update(obs)

        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps
        terminated = bool(ended)

        reward = self.alive_reward
        reason = ""
        if terminated:
            reward += self.death_penalty
            reason = "screen_static"
        elif truncated:
            reason = "time_limit"

        self.episode_reward += reward

        if terminated or truncated:
            self.controller.release()
            self._append_episode(reason, motion)

        info = {
            "episode_index": self.episode_index,
            "episode_steps": self.episode_steps,
            "survival_seconds": self.episode_steps / self.step_hz,
            "best_seconds": max(self.best_steps, self.episode_steps) / self.step_hz,
            "motion": motion,
            "termination_reason": reason or None,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self.controller.release()
        self.capture.close()
        self._closed = True
