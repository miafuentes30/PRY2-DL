"""Entrena QR-DQN en Space Invaders con recolección paralela."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from src.callbacks import CompetitionEvalCallback
from src.config import load_config
from src.double_qrdqn import DoubleQRDQN
from src.envs import make_space_invaders_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--resume-buffer", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    return parser.parse_args()


def configure_runtime(seed: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_runtime(config.experiment.seed)
    run_dir = Path("runs") / config.experiment.name
    for folder in (run_dir, run_dir / "monitor", run_dir / "evaluation", run_dir / "checkpoints"):
        folder.mkdir(parents=True, exist_ok=True)
    Path(run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "config_file": str(Path(args.config).resolve()),
                "seed": config.experiment.seed,
                "num_envs": config.environment.num_envs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_env = make_space_invaders_env(
        config.environment,
        seed=config.experiment.seed,
        training=True,
        monitor_dir=run_dir / "monitor",
    )
    eval_env = make_space_invaders_env(
        config.environment,
        seed=config.experiment.seed + 10_000,
        training=False,
        n_envs=1,
    )
    alg = config.algorithm
    if args.resume_model:
        model = DoubleQRDQN.load(args.resume_model, env=train_env, device=config.experiment.device)
        if args.resume_buffer:
            model.load_replay_buffer(args.resume_buffer)
        reset_num_timesteps = False
    else:
        model = DoubleQRDQN(
            "CnnPolicy",
            train_env,
            learning_rate=alg.learning_rate,
            buffer_size=alg.buffer_size,
            learning_starts=alg.learning_starts,
            batch_size=alg.batch_size,
            gamma=alg.gamma,
            train_freq=alg.train_freq,
            gradient_steps=alg.gradient_steps,
            n_steps=alg.n_steps,
            target_update_interval=alg.target_update_interval,
            exploration_fraction=alg.exploration_fraction,
            exploration_initial_eps=alg.exploration_initial_eps,
            exploration_final_eps=alg.exploration_final_eps,
            max_grad_norm=alg.max_grad_norm,
            optimize_memory_usage=alg.optimize_memory_usage,
            policy_kwargs={"n_quantiles": alg.n_quantiles},
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=1,
            seed=config.experiment.seed,
            device=config.experiment.device,
        )
        reset_num_timesteps = True

    # Los callbacks se invocan una vez por paso vectorizado, no por transición.
    eval_freq = max(config.evaluation.frequency // config.environment.num_envs, 1)
    checkpoint_freq = max(config.checkpoint.frequency // config.environment.num_envs, 1)
    callbacks = CallbackList(
        [
            CompetitionEvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=config.evaluation.episodes,
                output_dir=run_dir / "evaluation",
                deterministic=config.evaluation.deterministic,
            ),
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="qrdqn",
                save_replay_buffer=config.checkpoint.save_replay_buffer,
                save_vecnormalize=False,
            ),
        ]
    )
    try:
        model.learn(
            total_timesteps=args.timesteps or config.experiment.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=config.experiment.name,
        )
        model.save(run_dir / "final_model")
        if config.checkpoint.save_replay_buffer:
            model.save_replay_buffer(run_dir / "final_replay_buffer.pkl")
    finally:
        train_env.close()


if __name__ == "__main__":
    main()
