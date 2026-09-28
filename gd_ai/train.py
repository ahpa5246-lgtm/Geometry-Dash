from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .callbacks import BestEpisodeCallback
from .config import load_config
from .env import GeometryDashEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a pixel-only Geometry Dash agent from scratch."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override learning.total_timesteps.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing RecurrentPPO .zip file. Omit to start from random weights.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    learning = cfg["learning"]

    runs_dir = Path(paths["runs_dir"])
    models_dir = Path(paths["models_dir"])
    tensorboard_dir = Path(paths["tensorboard_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(GeometryDashEnv(cfg), filename=str(runs_dir / "monitor.csv"))
    total_timesteps = int(args.timesteps or learning["total_timesteps"])

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = RecurrentPPO.load(
            args.resume,
            env=env,
            device=str(learning.get("device", "auto")),
        )
    else:
        print("Starting from RANDOM weights. No dataset and no pretrained model.")
        model = RecurrentPPO(
            policy="CnnLstmPolicy",
            env=env,
            learning_rate=float(learning["learning_rate"]),
            n_steps=int(learning["n_steps"]),
            batch_size=int(learning["batch_size"]),
            gamma=float(learning["gamma"]),
            gae_lambda=float(learning["gae_lambda"]),
            ent_coef=float(learning["ent_coef"]),
            vf_coef=float(learning["vf_coef"]),
            clip_range=float(learning["clip_range"]),
            verbose=1,
            seed=int(cfg.get("seed", 42)),
            tensorboard_log=str(tensorboard_dir),
            device=str(learning.get("device", "auto")),
            policy_kwargs={
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
                "normalize_images": True,
            },
        )

    checkpoint = CheckpointCallback(
        save_freq=int(learning["checkpoint_every_steps"]),
        save_path=str(models_dir / "checkpoints"),
        name_prefix="gd_agent",
    )
    best = BestEpisodeCallback(models_dir)
    callbacks = CallbackList([checkpoint, best])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        path = models_dir / "interrupted_model"
        model.save(path)
        print(f"Training interrupted safely. Saved: {path}.zip")
    else:
        path = models_dir / "final_model"
        model.save(path)
        print(f"Training complete. Saved: {path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
