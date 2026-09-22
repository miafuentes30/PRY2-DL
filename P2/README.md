# Proyecto 2 - Agente Double QR-DQN para Space Invaders

Solución reproducible para `ALE/SpaceInvaders-v5`, orientada tanto al trabajo escrito como a la competencia. El agente combina:

- **QR-DQN**: aproxima una distribución de retornos con 200 cuantiles, no solo un valor Q medio.
- **Double DQN**: la red online selecciona la acción y la red objetivo la evalúa para reducir sobreestimación.
- **Retornos de 3 pasos**: propagan antes la recompensa sin abandonar el bootstrap.
- **CNN Nature** sobre cuatro frames grises de `84 x 84`.
- **Replay buffer**, red objetivo, clipping de gradiente y exploración epsilon-greedy.
- **8 entornos paralelos** con procesos separados y una inferencia batched.
- Selección del checkpoint por el **mejor de 5 episodios**, exactamente como la competencia; el promedio desempata.

No se incluye un modelo ya entrenado: los puntajes deben salir de una ejecución real y quedan registrados automáticamente.

## 1. Instalación en Windows 11

Se recomienda Python 3.12, una GPU NVIDIA y al menos 16 GB de RAM (32 GB es preferible para la configuración competitiva).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La primera línea de `requirements.txt` permite instalar PyTorch con CUDA 12.8. Si el equipo no tiene NVIDIA, se puede eliminar esa línea y reinstalar `torch` desde el índice normal.

Comprobar CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Prueba corta antes del entrenamiento real

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe train.py --config config_smoke.yaml
```

El smoke test verifica que las ROM, los wrappers, el multiprocesamiento, la GPU y el guardado funcionen. Sus puntajes no son representativos.

## 3. Entrenamiento competitivo

```powershell
.\.venv\Scripts\python.exe train.py --config config.yaml
```

Monitorear desde otra terminal:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs
```

Los archivos importantes quedan en:

- `runs/qrdqn_space_invaders_v1/evaluation/best_model.zip`: mejor checkpoint.
- `runs/qrdqn_space_invaders_v1/evaluation/best_metrics.json`: cinco puntajes que justifican la selección.
- `runs/qrdqn_space_invaders_v1/evaluation/evaluations.csv`: historial para el informe.
- `runs/qrdqn_space_invaders_v1/checkpoints/`: respaldos periódicos.
- `runs/qrdqn_space_invaders_v1/final_model.zip`: modelo del último paso (no necesariamente el mejor).

No se debe decidir por la recompensa de entrenamiento, porque allí las recompensas están recortadas a `[-1, 1]` y cada vida se trata como terminal para facilitar el aprendizaje. La evaluación usa vidas completas y puntaje real, sin clipping.

## 4. Evaluación final y video

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --model runs/qrdqn_space_invaders_v1/evaluation/best_model.zip `
  --episodes 5 `
  --seed 2026
```

Esto genera `videos/evaluacion_final/evaluation_results.json` y el MP4 de la partida con mayor puntaje. El script usa política greedy, las mismas dimensiones y el mismo frame stacking del entrenamiento.

## 5. Análisis y gráficas para el trabajo escrito

```powershell
.\.venv\Scripts\python.exe analyze_env.py --episodes 20
.\.venv\Scripts\python.exe plot_results.py
```

`analyze_env.py` documenta espacios, acciones, recompensas, vidas y baseline aleatorio. `plot_results.py` crea la curva del promedio y del mejor de cinco.

## 6. Reanudar una corrida

```powershell
.\.venv\Scripts\python.exe train.py `
  --resume-model runs/qrdqn_space_invaders_v1/checkpoints/qrdqn_2000000_steps.zip `
  --timesteps 2000000
```

Para reanudar exactamente el replay buffer, activar `save_replay_buffer: true`; requiere varios GB y checkpoints lentos. Luego agregar `--resume-buffer ruta_al_buffer.pkl`.

## Decisiones teóricas importantes

1. **Sin doble frameskip.** ALE se crea con `frameskip=1` y `AtariWrapper` aplica `frame_skip=4` una sola vez.
2. **Sticky actions de 0.25.** Aumentan robustez y coinciden con la configuración estándar del entorno v5 usada en el laboratorio.
3. **Espacio mínimo de acciones.** Evita acciones redundantes y reduce la dificultad de exploración.
4. **Sin reward shaping.** Se conserva el objetivo real. Solo se recorta la recompensa durante aprendizaje para estabilizar la escala.
5. **Evaluación separada.** No comparte estado con entrenamiento y reporta recompensas originales.
6. **Semillas distintas.** Entrenamiento, validación y evaluación final usan rangos diferentes para reducir selección por azar.

## Iteraciones recomendadas para el informe

| Iteración | Cambio aislado                               | Presupuesto sugerido |
| --------- | -------------------------------------------- | -------------------: |
| I0        | Agente aleatorio (`analyze_env.py`)          |         20 episodios |
| I1        | QR-DQN, 1-step, 50 cuantiles                 |     1 M transiciones |
| I2        | Double QR-DQN, 3-step, 200 cuantiles         |     3 M transiciones |
| I3        | Configuración final, entrenamiento extendido |    10 M transiciones |

Cambiar una sola familia de parámetros por iteración hace que la discusión sea defendible. Para I1 basta duplicar `config.yaml`, cambiar `n_steps: 1`, `n_quantiles: 50`, `num_envs: 4` y un nombre de experimento diferente.

## Uso de memoria

Las observaciones se almacenan como `uint8`. Con 100 000 transiciones y cuatro frames, el replay buffer puede ocupar aproximadamente 5-7 GB. Si el equipo tiene 16 GB de RAM, reducir `buffer_size` a `50000`. No aumentar los entornos pensando que siempre será más rápido: 4-8 suele ser un rango razonable y debe medirse en TensorBoard.
