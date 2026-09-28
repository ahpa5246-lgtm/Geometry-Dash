from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class BestEpisodeCallback(BaseCallback):
    """Save the model whenever a new longest episode is observed."""

    def __init__(self, save_dir: str | Path, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.save_dir = Path(save_dir)
        self.best_length = 0
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            length = int(episode.get("l", 0))
            if length > self.best_length:
                self.best_length = length
                path = self.save_dir / "best_model"
                self.model.save(path)
                if self.verbose:
                    print(f"New best episode: {length} steps -> {path}.zip")
        return True
