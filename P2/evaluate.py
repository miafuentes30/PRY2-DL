"""Evalúa cinco episodios greedy y graba el mejor episodio en MP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from src.config import load_config
from src.double_qrdqn import DoubleQRDQN
from src.envs import make_space_invaders_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ruta al best_model.zip")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--video-dir", default="videos/evaluacion_final")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def frame_from_env(env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, list):
        frame = frame[0]
    array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    return array.astype(np.uint8, copy=False)


def play_episode(model, env, *, seed: int, max_steps: int, capture: bool):
    env.seed(seed)
    obs = env.reset()
    total_reward, steps = 0.0, 0
    frames = [frame_from_env(env)] if capture else []
    done = np.array([False])
    while not bool(done[0]) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward[0])
        steps += 1
        if capture:
            frames.append(frame_from_env(env))
    return total_reward, steps, frames, bool(done[0])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.episodes < 1:
        raise ValueError("--episodes debe ser al menos 1.")
    env = make_space_invaders_env(
        config.environment, seed=args.seed, training=False, n_envs=1
    )
    model = DoubleQRDQN.load(args.model, device=config.experiment.device)
    results = []
    best_score = -np.inf
    best_seed = args.seed
    for episode in range(args.episodes):
        score, steps, _, natural_end = play_episode(
            model,
            env,
            seed=args.seed + episode,
            max_steps=config.evaluation.max_steps_per_episode,
            capture=False,
        )
        record = {
            "episode": episode + 1,
            "seed": args.seed + episode,
            "score": score,
            "steps": steps,
            "natural_end": natural_end,
        }
        results.append(record)
        print(record)
        if score > best_score:
            best_score, best_seed = score, args.seed + episode

    scores = np.asarray([row["score"] for row in results], dtype=np.float64)
    summary = {
        "model": str(Path(args.model).resolve()),
        "environment": config.environment.id,
        "episodes": results,
        "best_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
    }
    output_dir = Path(args.video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not args.no_video:
        video_score, _, best_frames, _ = play_episode(
            model,
            env,
            seed=best_seed,
            max_steps=config.evaluation.max_steps_per_episode,
            capture=True,
        )
        video_path = output_dir / "space_invaders_best_episode.mp4"
        imageio.mimsave(video_path, best_frames, fps=15, macro_block_size=1)
        summary["video"] = str(video_path)
        summary["video_seed"] = best_seed
        summary["video_replay_score"] = video_score
        (output_dir / "evaluation_results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Video guardado en {video_path}")
    env.close()
    print(f"Mejor de {args.episodes}: {scores.max():.1f}; promedio: {scores.mean():.1f}")


if __name__ == "__main__":
    main()
