from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import cv2

from .config import load_config
from .input import GameController
from .vision import MotionEndDetector
from .window import ScreenCapture


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Geometry Dash capture and estimate visual motion."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="How long to sample screen motion.",
    )
    parser.add_argument(
        "--test-input",
        action="store_true",
        help="Tap the configured action key once after capture.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    win = cfg["window"]
    env = cfg["environment"]
    inp = cfg["input"]

    capture = ScreenCapture(
        title_contains=str(win["title_contains"]),
        width=int(win["observation_width"]),
        height=int(win["observation_height"]),
        crop=win.get("crop"),
        grayscale=bool(win.get("grayscale", True)),
    )
    controller = GameController(
        action_key=str(inp.get("key", "space")),
        start_key=str(inp.get("start_key", "space")),
    )

    capture.focus()
    time.sleep(0.25)
    bgr = capture.grab_bgr()

    runs = Path(cfg["paths"]["runs_dir"])
    runs.mkdir(parents=True, exist_ok=True)
    image_path = runs / "calibration.png"
    cv2.imwrite(str(image_path), bgr)

    region = capture.region
    print(f"Window: {region.title}")
    print(f"Client area: {region.width}x{region.height} at ({region.left}, {region.top})")
    print(f"Saved capture: {image_path}")

    detector = MotionEndDetector(
        threshold=float(env["freeze_motion_threshold"]),
        freeze_frames=int(env["freeze_frames"]),
        warmup_frames=0,
    )

    hz = float(env["step_hz"])
    interval = 1.0 / hz
    samples: list[float] = []
    end_at = time.perf_counter() + max(0.5, args.seconds)

    print("Sampling motion. Keep the game visible...")
    while time.perf_counter() < end_at:
        obs = capture.grab_observation()
        _, motion = detector.update(obs)
        if motion <= 1.0:
            samples.append(motion)
        time.sleep(interval)

    if len(samples) > 2:
        samples = samples[1:]
        print(
            "Motion statistics: "
            f"median={statistics.median(samples):.6f}, "
            f"min={min(samples):.6f}, max={max(samples):.6f}"
        )
        print(
            "Configured freeze threshold: "
            f"{float(env['freeze_motion_threshold']):.6f}"
        )
        print(
            "During active gameplay, the median should be clearly ABOVE the "
            "freeze threshold. On a death/retry screen, it should usually be BELOW it."
        )

    if args.test_input:
        print("Testing one jump input...")
        capture.focus()
        controller.tap_test()

    controller.release()
    capture.close()
    print("Calibration finished.")


if __name__ == "__main__":
    main()
