"""Genera las curvas que se incorporarán al trabajo escrito."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/qrdqn_space_invaders_v1")
    args = parser.parse_args()
    run_dir = Path(args.run)
    eval_path = run_dir / "evaluation" / "evaluations.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"No se encontró {eval_path}")
    data = pd.read_csv(eval_path)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(data["timesteps"], data["mean_score"], label="Promedio (5 episodios)")
    axis.plot(data["timesteps"], data["max_score"], label="Mejor de 5")
    axis.set(title="Evaluación greedy de Double QR-DQN", xlabel="Transiciones", ylabel="Puntaje")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = run_dir / "evaluation" / "learning_curve.png"
    figure.savefig(output, dpi=180)
    print(f"Gráfica guardada en {output}")


if __name__ == "__main__":
    main()

