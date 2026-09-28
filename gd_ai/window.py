from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import mss
import numpy as np
import win32con
import win32gui


@dataclass(frozen=True)
class WindowRegion:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def _visible_windows() -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []

    def callback(hwnd: int, _: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if title:
            result.append((hwnd, title))

    win32gui.EnumWindows(callback, None)
    return result


def find_window(title_contains: str) -> WindowRegion:
    needle = title_contains.casefold()
    matches = [(hwnd, title) for hwnd, title in _visible_windows() if needle in title.casefold()]
    if not matches:
        titles = "\n".join(f"  - {title}" for _, title in _visible_windows()[:30])
        raise RuntimeError(
            f'No visible window contains "{title_contains}".\n'
            f"Open Geometry Dash first. Visible windows include:\n{titles}"
        )

    hwnd, title = matches[0]
    return client_region(hwnd, title)


def client_region(hwnd: int, title: str | None = None) -> WindowRegion:
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
    width = max(1, right - left)
    height = max(1, bottom - top)
    return WindowRegion(
        hwnd=hwnd,
        title=title or win32gui.GetWindowText(hwnd),
        left=screen_left,
        top=screen_top,
        width=width,
        height=height,
    )


def focus_window(hwnd: int) -> None:
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.15)
    try:
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        # Windows can block focus-stealing in some situations. Input may still work
        # if the user left the game focused.
        pass


class ScreenCapture:
    def __init__(
        self,
        title_contains: str,
        width: int,
        height: int,
        crop: dict[str, float] | None = None,
        grayscale: bool = True,
    ) -> None:
        self.title_contains = title_contains
        self.obs_width = int(width)
        self.obs_height = int(height)
        self.crop = crop or {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}
        self.grayscale = bool(grayscale)
        self._sct = mss.mss()
        self.region = find_window(title_contains)

    def refresh_region(self) -> WindowRegion:
        self.region = client_region(self.region.hwnd, self.region.title)
        return self.region

    def focus(self) -> None:
        focus_window(self.region.hwnd)

    def _monitor(self) -> dict[str, int]:
        r = self.refresh_region()
        x0 = int(r.left + r.width * float(self.crop["left"]))
        y0 = int(r.top + r.height * float(self.crop["top"]))
        x1 = int(r.left + r.width * float(self.crop["right"]))
        y1 = int(r.top + r.height * float(self.crop["bottom"]))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid crop values in config.yaml")
        return {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}

    def grab_bgr(self) -> np.ndarray:
        shot = np.asarray(self._sct.grab(self._monitor()), dtype=np.uint8)
        return shot[:, :, :3]

    def grab_observation(self) -> np.ndarray:
        frame = self.grab_bgr()
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
            return frame[:, :, None].astype(np.uint8, copy=False)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8, copy=False)

    def close(self) -> None:
        self._sct.close()
