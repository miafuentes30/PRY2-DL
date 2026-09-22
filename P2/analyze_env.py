"""Genera el EDA mínimo del entorno solicitado en el trabajo escrito."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np


gym.register_envs(ale_py)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/environment_analysis.json")
    args = parser.parse_args()
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        frameskip=4,
        repeat_action_probability=0.25,
        full_action_space=False,
        obs_type="rgb",
    )
    scores, lengths, rewards = [], [], Counter()
    lives_seen = set()
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed + episode)
            env.action_space.seed(args.seed + episode)
            done, score, length = False, 0.0, 0
            while not done:
                _, reward, terminated, truncated, info = env.step(env.action_space.sample())
                score += float(reward)
                length += 1
                rewards[str(float(reward))] += 1
                if "lives" in info:
                    lives_seen.add(int(info["lives"]))
                done = terminated or truncated
            scores.append(score)
            lengths.append(length)
        result = {
            "environment": "ALE/SpaceInvaders-v5",
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "action_meanings": env.unwrapped.get_action_meanings(),
            "episodes": args.episodes,
            "random_score_mean": float(np.mean(scores)),
            "random_score_std": float(np.std(scores)),
            "random_score_max": float(np.max(scores)),
            "episode_length_mean": float(np.mean(lengths)),
            "reward_frequencies": dict(rewards),
            "lives_seen": sorted(lives_seen),
        }
    finally:
        env.close()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

