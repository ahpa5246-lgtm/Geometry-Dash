from __future__ import annotations

import time

import pydirectinput


pydirectinput.PAUSE = 0


class GameController:
    """Only exposes the actions the agent is allowed to learn: release or hold."""

    def __init__(self, action_key: str = "space", start_key: str = "space") -> None:
        self.action_key = action_key
        self.start_key = start_key
        self._held = False

    def set_action(self, action: int) -> None:
        if int(action) == 1 and not self._held:
            pydirectinput.keyDown(self.action_key)
            self._held = True
        elif int(action) == 0 and self._held:
            pydirectinput.keyUp(self.action_key)
            self._held = False

    def release(self) -> None:
        if self._held:
            pydirectinput.keyUp(self.action_key)
            self._held = False

    def restart(self) -> None:
        self.release()
        pydirectinput.press(self.start_key)

    def tap_test(self, seconds: float = 0.08) -> None:
        self.release()
        pydirectinput.keyDown(self.action_key)
        time.sleep(seconds)
        pydirectinput.keyUp(self.action_key)
