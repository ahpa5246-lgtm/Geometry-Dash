from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class MotionEndDetector:
    """Detects a likely death/end screen without labels or templates.

    Geometry Dash continuously scrolls during a run. Once the run ends, most of
    the client image becomes nearly static. We measure normalized frame change
    and terminate after several consecutive low-motion frames.
    """

    def __init__(
        self,
        threshold: float = 0.004,
        freeze_frames: int = 8,
        warmup_frames: int = 18,
    ) -> None:
        self.threshold = float(threshold)
        self.freeze_frames = int(freeze_frames)
        self.warmup_frames = int(warmup_frames)
        self._prev: np.ndarray | None = None
        self._low_motion = 0
        self._steps = 0
        self.motion_history: deque[float] = deque(maxlen=120)

    def reset(self) -> None:
        self._prev = None
        self._low_motion = 0
        self._steps = 0
        self.motion_history.clear()

    @staticmethod
    def _small_gray(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[-1] == 1:
            gray = obs[:, :, 0]
        elif obs.ndim == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return cv2.resize(gray, (42, 42), interpolation=cv2.INTER_AREA)

    def update(self, obs: np.ndarray) -> tuple[bool, float]:
        current = self._small_gray(obs)
        self._steps += 1

        if self._prev is None:
            self._prev = current
            return False, 1.0

        delta = cv2.absdiff(current, self._prev)
        motion = float(delta.mean() / 255.0)
        self.motion_history.append(motion)
        self._prev = current

        if self._steps <= self.warmup_frames:
            self._low_motion = 0
            return False, motion

        if motion < self.threshold:
            self._low_motion += 1
        else:
            self._low_motion = 0

        return self._low_motion >= self.freeze_frames, motion
