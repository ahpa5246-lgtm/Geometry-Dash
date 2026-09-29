from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import load_config
from .env import GeometryDashEnv
from .offline_q import PixelQNetwork
from .offline_train import choose_device


def to_gray(obs: np.ndarray) -> np.ndarray:
    if obs.shape[-1] == 1:
        return obs[..., 0]
    return (
        0.299 * obs[..., 0]
        + 0.587 * obs[..., 1]
        + 0.114 * obs[..., 2]
    ).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the offline-trained visual Q policy in the live game."
    )
    parser.add_argument("model", help="Path to offline_q_*.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(str(cfg["offline_learning"].get("device", "auto")))

    checkpoint = torch.load(args.model, map_location=device)
    model = PixelQNetwork().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    env = GeometryDashEnv(cfg)
    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            prev = to_gray(obs)
            done = False
            info = {}

            while not done:
                current = to_gray(obs)
                state = np.stack([prev, current], axis=0)[None, ...]
                state_t = torch.from_numpy(state).to(device=device, dtype=torch.float32).div_(255.0)

                with torch.no_grad():
                    action = int(model(state_t).argmax(dim=1).item())

                prev = current
                obs, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

            print(
                f"Episode {episode}: "
                f"{float(info.get('survival_seconds', 0.0)):.2f}s "
                f"({info.get('termination_reason')})"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
