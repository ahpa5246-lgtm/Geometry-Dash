from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .config import load_config
from .offline_q import DiskReplaySampler, PixelQNetwork


def choose_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continue learning while Geometry Dash is CLOSED, using only replay "
            "that the agent previously generated itself."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Optional offline Q checkpoint (.pt).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    offline = cfg["offline_learning"]
    paths = cfg["paths"]

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device(str(offline.get("device", "auto")))
    sampler = DiskReplaySampler(paths["replay_dir"])
    print(f"Offline replay transitions available: {sampler.transition_count():,}")
    print(f"Training device: {device}")
    print("Geometry Dash does NOT need to be running for this command.")

    online = PixelQNetwork().to(device)
    target = PixelQNetwork().to(device)
    optimizer = torch.optim.Adam(
        online.parameters(),
        lr=float(offline["learning_rate"]),
    )

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        online.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint.get("target_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(f"Resumed offline learner from step {start_step:,}")
    else:
        target.load_state_dict(online.state_dict())
        print("Offline Q-network starts from RANDOM weights.")

    target.eval()

    total_steps = int(args.steps or offline["gradient_steps"])
    batch_size = int(offline["batch_size"])
    gamma = float(offline["gamma"])
    cql_alpha = float(offline["cql_alpha"])
    target_update = int(offline["target_update_every"])
    save_every = int(offline["save_every"])

    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    rolling_loss = 0.0
    rolling_td = 0.0
    rolling_cql = 0.0

    try:
        for local_step in range(1, total_steps + 1):
            step = start_step + local_step
            states, actions, rewards, next_states, dones = sampler.sample(batch_size, device)

            q_all = online(states)
            q_data = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_action = online(next_states).argmax(dim=1, keepdim=True)
                next_q = target(next_states).gather(1, next_action).squeeze(1)
                td_target = rewards + gamma * (1.0 - dones) * next_q

            td_loss = F.smooth_l1_loss(q_data, td_target)

            # Discrete Conservative Q-Learning penalty. It discourages high values
            # for actions that are unsupported by the agent's own replay.
            cql_loss = (torch.logsumexp(q_all, dim=1) - q_data).mean()
            loss = td_loss + cql_alpha * cql_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            optimizer.step()

            rolling_loss += float(loss.item())
            rolling_td += float(td_loss.item())
            rolling_cql += float(cql_loss.item())

            if step % target_update == 0:
                target.load_state_dict(online.state_dict())

            if local_step % 100 == 0:
                print(
                    f"offline step {step:,} | "
                    f"loss={rolling_loss / 100:.5f} | "
                    f"td={rolling_td / 100:.5f} | "
                    f"cql={rolling_cql / 100:.5f}"
                )
                rolling_loss = rolling_td = rolling_cql = 0.0

            if step % save_every == 0:
                path = models_dir / f"offline_q_{step:09d}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": online.state_dict(),
                        "target_model": target.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    path,
                )
                print(f"Saved offline checkpoint: {path}")

    except KeyboardInterrupt:
        print("Offline training interrupted safely.")

    final_step = start_step + local_step if "local_step" in locals() else start_step
    final_path = models_dir / "offline_q_latest.pt"
    torch.save(
        {
            "step": final_step,
            "model": online.state_dict(),
            "target_model": target.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        final_path,
    )
    print(f"Saved latest offline learner: {final_path}")


if __name__ == "__main__":
    main()
