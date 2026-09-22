"""Carga y validación de la configuración del experimento."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    total_timesteps: int
    device: str


@dataclass(frozen=True)
class EnvironmentConfig:
    id: str
    num_envs: int
    frame_skip: int
    frame_stack: int
    screen_size: int
    repeat_action_probability: float
    terminal_on_life_loss_train: bool
    clip_reward_train: bool
    full_action_space: bool


@dataclass(frozen=True)
class AlgorithmConfig:
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    gamma: float
    train_freq: int
    gradient_steps: int
    n_steps: int
    target_update_interval: int
    exploration_fraction: float
    exploration_initial_eps: float
    exploration_final_eps: float
    n_quantiles: int
    max_grad_norm: float
    optimize_memory_usage: bool


@dataclass(frozen=True)
class EvaluationConfig:
    frequency: int
    episodes: int
    max_steps_per_episode: int
    deterministic: bool


@dataclass(frozen=True)
class CheckpointConfig:
    frequency: int
    save_replay_buffer: bool


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig


def _section(data: dict[str, Any], name: str, cls: type[Any]) -> Any:
    if name not in data or not isinstance(data[name], dict):
        raise ValueError(f"Falta la sección '{name}' en la configuración.")
    try:
        return cls(**data[name])
    except TypeError as exc:
        raise ValueError(f"La sección '{name}' tiene campos inválidos: {exc}") from exc


def load_config(path: str | Path = "config.yaml") -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("La raíz de config.yaml debe ser un objeto.")
    config = Config(
        experiment=_section(raw, "experiment", ExperimentConfig),
        environment=_section(raw, "environment", EnvironmentConfig),
        algorithm=_section(raw, "algorithm", AlgorithmConfig),
        evaluation=_section(raw, "evaluation", EvaluationConfig),
        checkpoint=_section(raw, "checkpoint", CheckpointConfig),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    env, alg, ev = config.environment, config.algorithm, config.evaluation
    positive = {
        "total_timesteps": config.experiment.total_timesteps,
        "num_envs": env.num_envs,
        "frame_skip": env.frame_skip,
        "frame_stack": env.frame_stack,
        "screen_size": env.screen_size,
        "buffer_size": alg.buffer_size,
        "learning_starts": alg.learning_starts,
        "batch_size": alg.batch_size,
        "train_freq": alg.train_freq,
        "gradient_steps": alg.gradient_steps,
        "n_steps": alg.n_steps,
        "target_update_interval": alg.target_update_interval,
        "n_quantiles": alg.n_quantiles,
        "evaluation.episodes": ev.episodes,
        "evaluation.frequency": ev.frequency,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} debe ser mayor que cero.")
    if not 0.0 < alg.gamma <= 1.0:
        raise ValueError("gamma debe pertenecer a (0, 1].")
    if not 0.0 <= env.repeat_action_probability <= 1.0:
        raise ValueError("repeat_action_probability debe pertenecer a [0, 1].")
    if not 0.0 <= alg.exploration_final_eps <= alg.exploration_initial_eps <= 1.0:
        raise ValueError("Los valores epsilon deben cumplir 0 <= final <= inicial <= 1.")
    if alg.learning_starts >= config.experiment.total_timesteps:
        raise ValueError("learning_starts debe ser menor que total_timesteps.")

