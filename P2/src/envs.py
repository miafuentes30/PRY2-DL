"""Construcción centralizada de entornos Atari para entrenamiento y evaluación."""
from __future__ import annotations

from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from src.config import EnvironmentConfig


gym.register_envs(ale_py)


def make_space_invaders_env(
    config: EnvironmentConfig,
    *,
    seed: int,
    training: bool,
    monitor_dir: str | Path | None = None,
    n_envs: int | None = None,
):
    """Crea un VecEnv con preprocessing idéntico salvo recompensas y vidas.

    El ALE interno usa frameskip=1 porque AtariWrapper realiza el salto de
    frames configurado. Esto evita aplicar el salto dos veces.
    """
    workers = config.num_envs if n_envs is None else n_envs
    if workers < 1:
        raise ValueError("n_envs debe ser al menos 1.")
    vec_cls = SubprocVecEnv if workers > 1 else DummyVecEnv
    vec_kwargs = {"start_method": "spawn"} if workers > 1 else None
    env = make_atari_env(
        config.id,
        n_envs=workers,
        seed=seed,
        monitor_dir=None if monitor_dir is None else str(monitor_dir),
        env_kwargs={
            "frameskip": 1,
            "repeat_action_probability": config.repeat_action_probability,
            "full_action_space": config.full_action_space,
        },
        wrapper_kwargs={
            "screen_size": config.screen_size,
            "frame_skip": config.frame_skip,
            "terminal_on_life_loss": (
                config.terminal_on_life_loss_train if training else False
            ),
            "clip_reward": config.clip_reward_train if training else False,
        },
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
    )
    return VecFrameStack(env, n_stack=config.frame_stack)

