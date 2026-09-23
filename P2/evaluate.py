"""Evalúa episodios greedy, registra las acciones y graba el mejor episodio en MP4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from src.config import load_config
from src.double_qrdqn import DoubleQRDQN
from src.envs import make_space_invaders_env


# ============================================================
# ACCIONES DE SPACE INVADERS
# ============================================================

ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}

# Estas son las acciones en las que LA NAVE dispara.
SHOOTING_ACTIONS = {1, 4, 5}


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        required=True,
        help="Ruta al best_model.zip",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--video-dir",
        default="videos/evaluacion_final",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="No generar video.",
    )

    parser.add_argument(
        "--show-shots",
        action="store_true",
        help=(
            "Muestra en consola cada vez que el agente "
            "ejecuta una acción de disparo."
        ),
    )

    return parser.parse_args()


# ============================================================
# OBTENER FRAME PARA VIDEO
# ============================================================

def frame_from_env(env) -> np.ndarray:
    frame = env.render()

    if isinstance(frame, list):
        frame = frame[0]

    array = np.asarray(frame)

    if array.ndim == 4:
        array = array[0]

    return array.astype(np.uint8, copy=False)


# ============================================================
# JUGAR UN EPISODIO
# ============================================================

def play_episode(
    model,
    env,
    *,
    seed: int,
    max_steps: int,
    capture: bool,
    show_shots: bool = False,
    episode_number: int | None = None,
):
    env.seed(seed)

    obs = env.reset()

    total_reward = 0.0
    steps = 0

    frames = [frame_from_env(env)] if capture else []

    done = np.array([False])

    action_counts = Counter()

    shot_number = 0

    while not bool(done[0]) and steps < max_steps:

        # ----------------------------------------------------
        # EL AGENTE ELIGE UNA ACCIÓN
        # ----------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action_id = int(
            np.asarray(action).reshape(-1)[0]
        )

        action_counts[action_id] += 1

        # ----------------------------------------------------
        # DETECTAR SI LA NAVE ESTÁ DISPARANDO
        # ----------------------------------------------------

        if action_id in SHOOTING_ACTIONS:
            shot_number += 1

            if show_shots:
                action_name = ACTION_NAMES.get(
                    action_id,
                    str(action_id),
                )

                episode_text = (
                    f"Episodio {episode_number}"
                    if episode_number is not None
                    else "Episodio"
                )

                print(
                    f"[DISPARO NAVE #{shot_number:04d}] "
                    f"{episode_text} | "
                    f"Paso {steps + 1:04d} | "
                    f"Acción: {action_name}"
                )

        # ----------------------------------------------------
        # EJECUTAR ACCIÓN EN SPACE INVADERS
        # ----------------------------------------------------

        obs, reward, done, _ = env.step(action)

        total_reward += float(reward[0])

        steps += 1

        # ----------------------------------------------------
        # GUARDAR FRAME PARA VIDEO
        # ----------------------------------------------------

        if capture:
            frames.append(
                frame_from_env(env)
            )

    return (
        total_reward,
        steps,
        frames,
        bool(done[0]),
        action_counts,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    config = load_config(args.config)

    if args.episodes < 1:
        raise ValueError(
            "--episodes debe ser al menos 1."
        )

    # --------------------------------------------------------
    # CREAR ENTORNO
    # --------------------------------------------------------

    env = make_space_invaders_env(
        config.environment,
        seed=args.seed,
        training=False,
        n_envs=1,
    )

    # --------------------------------------------------------
    # CARGAR MODELO
    # --------------------------------------------------------

    model = DoubleQRDQN.load(
        args.model,
        device=config.experiment.device,
    )

    results = []

    best_score = -np.inf
    best_seed = args.seed

    # ========================================================
    # EVALUACIÓN
    # ========================================================

    for episode in range(args.episodes):

        episode_number = episode + 1
        episode_seed = args.seed + episode

        print()
        print("=" * 70)
        print(
            f"EPISODIO {episode_number} "
            f"| Seed: {episode_seed}"
        )
        print("=" * 70)

        score, steps, _, natural_end, action_counts = play_episode(
            model,
            env,
            seed=episode_seed,
            max_steps=config.evaluation.max_steps_per_episode,
            capture=False,
            show_shots=args.show_shots,
            episode_number=episode_number,
        )

        # ----------------------------------------------------
        # CUÁNTAS ACCIONES DISPARAN
        # ----------------------------------------------------

        shooting_actions = (
            action_counts[1]
            + action_counts[4]
            + action_counts[5]
        )

        shooting_rate = (
            shooting_actions / steps * 100
            if steps > 0
            else 0.0
        )

        # ----------------------------------------------------
        # GUARDAR CONTEO DE ACCIONES
        # ----------------------------------------------------

        action_counts_named = {}

        for action_id, count in sorted(
            action_counts.items()
        ):
            action_name = ACTION_NAMES.get(
                action_id,
                str(action_id),
            )

            action_counts_named[action_name] = int(count)

        record = {
            "episode": episode_number,
            "seed": episode_seed,
            "score": float(score),
            "steps": int(steps),
            "natural_end": bool(natural_end),
            "shooting_actions": int(shooting_actions),
            "shooting_rate": float(shooting_rate),
            "action_counts": action_counts_named,
        }

        results.append(record)

        # ----------------------------------------------------
        # RESULTADOS DEL EPISODIO
        # ----------------------------------------------------

        print()
        print("RESULTADO DEL EPISODIO")
        print("-" * 70)

        print(
            f"Puntaje: {score:.1f}"
        )

        print(
            f"Pasos: {steps}"
        )

        print(
            f"Terminación natural: {natural_end}"
        )

        print()
        print("ACCIONES UTILIZADAS")
        print("-" * 70)

        for action_id in range(6):

            count = action_counts[action_id]

            action_name = ACTION_NAMES[action_id]

            percentage = (
                count / steps * 100
                if steps > 0
                else 0.0
            )

            print(
                f"{action_id} - "
                f"{action_name:<10} : "
                f"{count:5d} veces "
                f"({percentage:6.2f}%)"
            )

        print()
        print(
            "DISPAROS DE LA NAVE: "
            f"{shooting_actions}"
        )

        print(
            "PORCENTAJE DE ACCIONES "
            f"CON DISPARO: {shooting_rate:.2f}%"
        )

        # ----------------------------------------------------
        # MEJOR EPISODIO
        # ----------------------------------------------------

        if score > best_score:

            best_score = score
            best_seed = episode_seed

    # ========================================================
    # ESTADÍSTICAS GLOBALES
    # ========================================================

    scores = np.asarray(
        [row["score"] for row in results],
        dtype=np.float64,
    )

    total_steps = sum(
        row["steps"]
        for row in results
    )

    total_shooting_actions = sum(
        row["shooting_actions"]
        for row in results
    )

    overall_shooting_rate = (
        total_shooting_actions
        / total_steps
        * 100
        if total_steps > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("RESUMEN GLOBAL")
    print("=" * 70)

    print(
        f"Mejor puntaje: {scores.max():.1f}"
    )

    print(
        f"Puntaje promedio: {scores.mean():.1f}"
    )

    print(
        f"Desviación estándar: {scores.std():.2f}"
    )

    print(
        f"Total de pasos: {total_steps}"
    )

    print(
        "Total de acciones con disparo: "
        f"{total_shooting_actions}"
    )

    print(
        "Porcentaje global de acciones "
        f"que disparan: {overall_shooting_rate:.2f}%"
    )

    print(
        f"Seed del mejor episodio: {best_seed}"
    )

    # ========================================================
    # GUARDAR JSON
    # ========================================================

    summary = {
        "model": str(
            Path(args.model).resolve()
        ),
        "environment": config.environment.id,
        "episodes": results,
        "best_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "total_steps": int(total_steps),
        "total_shooting_actions": int(
            total_shooting_actions
        ),
        "overall_shooting_rate": float(
            overall_shooting_rate
        ),
        "best_seed": int(best_seed),
    }

    output_dir = Path(args.video_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "evaluation_results.json"
    )

    results_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Resultados guardados en: "
        f"{results_path}"
    )

    # ========================================================
    # GENERAR VIDEO DEL MEJOR EPISODIO
    # ========================================================

    if not args.no_video:

        print()
        print("=" * 70)
        print(
            "GENERANDO VIDEO DEL "
            "MEJOR EPISODIO"
        )
        print("=" * 70)

        (
            video_score,
            _,
            best_frames,
            _,
            _,
        ) = play_episode(
            model,
            env,
            seed=best_seed,
            max_steps=(
                config.evaluation
                .max_steps_per_episode
            ),
            capture=True,

            # No volver a imprimir cientos
            # de disparos durante el video.
            show_shots=False,
        )

        video_path = (
            output_dir
            / "space_invaders_best_episode.mp4"
        )

        imageio.mimsave(
            video_path,
            best_frames,
            fps=15,
            macro_block_size=1,
        )

        summary["video"] = str(
            video_path
        )

        summary["video_seed"] = int(
            best_seed
        )

        summary[
            "video_replay_score"
        ] = float(video_score)

        results_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"Video guardado en: "
            f"{video_path}"
        )

    # ========================================================
    # CERRAR ENTORNO
    # ========================================================

    env.close()

    print()
    print("=" * 70)
    print("EVALUACIÓN FINALIZADA")
    print("=" * 70)

    print(
        f"Mejor de {args.episodes}: "
        f"{scores.max():.1f}"
    )

    print(
        f"Promedio: "
        f"{scores.mean():.1f}"
    )


if __name__ == "__main__":
    main()