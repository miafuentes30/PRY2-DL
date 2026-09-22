"""Callbacks orientados a la regla real de la competencia."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class CompetitionEvalCallback(BaseCallback):
    """Evalúa cinco partidas y guarda por máximo; el promedio desempata."""

    def __init__(
        self,
        eval_env,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        output_dir: str | Path,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = n_eval_episodes
        self.output_dir = Path(output_dir)
        self.deterministic = deterministic
        self.best_key = (-np.inf, -np.inf)

    def _init_callback(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "evaluations.csv"
        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    ["timesteps", "scores", "max_score", "mean_score", "std_score"]
                )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        rewards, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=False,
        )
        scores = np.asarray(rewards, dtype=np.float64)
        current_key = (float(scores.max()), float(scores.mean()))
        record = {
            "timesteps": int(self.num_timesteps),
            "scores": scores.tolist(),
            "max_score": current_key[0],
            "mean_score": current_key[1],
            "std_score": float(scores.std()),
        }
        with (self.output_dir / "evaluations.csv").open(
            "a", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow(
                [record["timesteps"], json.dumps(record["scores"]), *current_key, record["std_score"]]
            )
        self.logger.record("competition/max_score", current_key[0])
        self.logger.record("competition/mean_score", current_key[1])
        if self.verbose:
            print(
                f"Evaluación @ {self.num_timesteps}: {scores.tolist()} | "
                f"máximo={current_key[0]:.1f}, promedio={current_key[1]:.1f}"
            )
        if current_key > self.best_key:
            self.best_key = current_key
            self.model.save(self.output_dir / "best_model")
            (self.output_dir / "best_metrics.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if self.verbose:
                print("Nuevo mejor modelo guardado según la regla de competencia.")
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()

