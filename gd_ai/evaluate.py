from __future__ import annotations

import argparse
import numpy as np
from sb3_contrib import RecurrentPPO

from .config import load_config
from .env import GeometryDashEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained model without learning.")
    parser.add_argument("model", help="Path to a RecurrentPPO .zip model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = GeometryDashEnv(cfg)
    model = RecurrentPPO.load(
        args.model,
        device=str(cfg["learning"].get("device", "auto")),
    )

    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    try:
        for index in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done = False
            lstm_state = None
            episode_start[:] = True
            last_info = {}

            while not done:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                obs, _, terminated, truncated, last_info = env.step(int(action.item()))
                done = terminated or truncated
                episode_start[:] = done

            print(
                f"Episode {index}: "
                f"{last_info.get('survival_seconds', 0.0):.2f}s "
                f"({last_info.get('termination_reason')})"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
