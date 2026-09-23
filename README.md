# PRY2-DL

Implementacion reproducible de un agente Double QR-DQN para `ALE/SpaceInvaders-v5`. El agente usa 200 cuantiles, seleccion de acciones Double DQN, retornos de 3 pasos, una CNN tipo Nature, cuatro frames apilados y cuatro entornos paralelos.

La configuracion final del proyecto es `config_v2.yaml`. El entrenamiento y la evaluacion deben ejecutarse desde esta carpeta (`P2/`). Los comandos siguientes usan PowerShell en Windows.

## 1. Instalar el entorno

Se recomienda Python 3.12. La instalacion incluye PyTorch con CUDA 12.8, ademas de Gymnasium/ALE, Stable-Baselines3 y las herramientas de analisis.

```powershell
cd P2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Comprobar que PyTorch detecta la GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Si no hay una GPU, se puede instalar la version de PyTorch para CPU quitando temporalmente `--extra-index-url` del archivo `requirements.txt` y reinstalando `torch` desde PyPI.

## 2. Verificar la instalacion

Primero ejecutar las pruebas unitarias:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Despues ejecutar el smoke test. Es una corrida corta para verificar ALE, los wrappers, el multiprocesamiento y el guardado; sus resultados no son representativos:

```powershell
.\.venv\Scripts\python.exe train.py --config config_smoke.yaml
```

## 3. Reproducir el entrenamiento final

La corrida final usa la semilla `123`, cuatro entornos paralelos y 15 000 000 transiciones:

```powershell
.\.venv\Scripts\python.exe train.py `
  --config config_v2.yaml `
  --timesteps 15000000
```

El argumento `--timesteps` es opcional: si se omite, se utiliza el valor de `experiment.total_timesteps` del archivo de configuracion.

Durante la corrida se crean los siguientes archivos en `runs/qrdqn_space_invaders_v2/`:

- `final_model.zip`: pesos del modelo del ultimo paso del entrenamiento.
- `evaluation/best_model.zip`: checkpoint con el mejor promedio de cinco episodios de validacion.
- `evaluation/best_metrics.json`: metricas de la evaluacion que produjo el mejor checkpoint.
- `evaluation/evaluations.csv`: historial de las evaluaciones.
- `checkpoints/qrdqn_*_steps.zip`: respaldos periodicos.
- `monitor/`: estadisticas de los entornos de entrenamiento.
- `tensorboard/`: registros para TensorBoard.
- `resolved_config.json`: semilla, configuracion usada y numero de entornos.

El checkpoint recomendado para reportar resultados es `evaluation/best_model.zip`. `final_model.zip` solo representa el ultimo estado y puede tener un rendimiento menor.

Para visualizar la corrida mientras entrena, abrir otra terminal en `P2/`:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs
```

## 4. Evaluar y generar el video

Evaluar el mejor checkpoint en cinco episodios, usando politica greedy y las recompensas originales:

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --config config_v2.yaml `
  --model runs/qrdqn_space_invaders_v2/evaluation/best_model.zip `
  --episodes 5 `
  --seed 2026
```

Para evaluar especificamente los pesos del ultimo paso, cambiar la ruta:

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --config config_v2.yaml `
  --model runs/qrdqn_space_invaders_v2/final_model.zip `
  --episodes 5 `
  --seed 2026
```

La evaluacion guarda:

- `videos/evaluacion_final/evaluation_results.json`: puntajes por episodio, promedio, desviacion estandar, acciones y semilla del mejor episodio.
- `videos/evaluacion_final/space_invaders_best_episode.mp4`: video del mejor episodio, salvo que se use `--no-video`.

El entorno de evaluacion no termina una partida al perder una vida y no aplica clipping a las recompensas. El entrenamiento si usa terminal por perdida de vida y clipping a `[-1, 1]` para estabilizar el aprendizaje.

## 5. Cargar los pesos desde Python

El archivo `.zip` contiene los pesos y la configuracion serializada de Stable-Baselines3. Para cargar el modelo final:

```python
from src.double_qrdqn import DoubleQRDQN

model = DoubleQRDQN.load(
    "runs/qrdqn_space_invaders_v2/final_model.zip",
    device="auto",
)
```

Para usarlo con un entorno y obtener una accion:

```python
from src.config import load_config
from src.envs import make_space_invaders_env

config = load_config("config_v2.yaml")
env = make_space_invaders_env(
    config.environment,
    seed=2026,
    training=False,
    n_envs=1,
)

observation = env.reset()
action, _ = model.predict(observation, deterministic=True)
observation, reward, done, info = env.step(action)
env.close()
```

Para cargar el mejor checkpoint, sustituir `final_model.zip` por `evaluation/best_model.zip`. La arquitectura, el numero de cuantiles y el preprocesamiento deben coincidir con los usados al entrenar; por eso se recomienda mantener `--config config_v2.yaml` en la evaluacion.

## 6. Reanudar un entrenamiento

Los checkpoints periodicos se cargan con `--resume-model`:

```powershell
.\.venv\Scripts\python.exe train.py `
  --config config_v2.yaml `
  --resume-model runs/qrdqn_space_invaders_v2/checkpoints/qrdqn_500000_steps.zip `
  --timesteps 2000000
```

La configuracion final no guarda el replay buffer porque ocupa varios GB. Para reanudar tambien el buffer, activar `checkpoint.save_replay_buffer: true` en una copia de la configuracion y pasar `--resume-buffer ruta_al_buffer.pkl`.

## 7. Analisis y graficas

```powershell
.\.venv\Scripts\python.exe analyze_env.py --episodes 20
.\.venv\Scripts\python.exe plot_results.py
```

`analyze_env.py` describe el espacio de acciones, observaciones, recompensas, vidas y baseline aleatorio. `plot_results.py` genera graficas a partir de los registros de evaluacion.

## 8. Configuracion y decisiones principales

- Entorno: `ALE/SpaceInvaders-v5`, con `frame_skip=4`, `frame_stack=4` y observaciones de `84 x 84` en escala de grises.
- Acciones: espacio reducido a `NOOP`, `FIRE`, `RIGHT`, `LEFT`, `RIGHTFIRE` y `LEFTFIRE`.
- Sticky actions: probabilidad `0.25`.
- Double DQN: la red online selecciona la accion y la red objetivo evalua sus cuantiles.
- Sin reward shaping: solo se recorta la recompensa durante el entrenamiento.
- Evaluacion: cinco episodios deterministas; el mejor checkpoint se selecciona por promedio, con el mejor puntaje como desempate.

La configuracion historica v1 se puede reproducir con `config.yaml` y genera sus resultados en `runs/qrdqn_space_invaders_v1/`:

```powershell
.\.venv\Scripts\python.exe train.py --config config.yaml
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
