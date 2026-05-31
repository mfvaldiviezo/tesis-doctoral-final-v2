"""
train.py — Entrenamiento PPO para Control Semafórico con Equidad
================================================================
Módulo 3: RL Agent — Script principal de entrenamiento.

Pipeline:
    1. Instancia N entornos en paralelo (SubprocVecEnv) con TrafficEnv.
    2. Registra métricas de equidad (Gini) y eficiencia (demora) en TensorBoard.
    3. Entrena un agente PPO con política MlpPolicy.
    4. Guarda checkpoints intermedios y el modelo final.

Uso:
    python src/rl_agent/train.py [--timesteps 500000] [--n-envs 4] [--seed 42]

Monitoreo en tiempo real:
    tensorboard --logdir tensorboard_logs/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Entorno personalizado
from src.rl_agent.sumo_env import TrafficEnv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"
_TB_LOG_DIR = _PROJECT_ROOT / "tensorboard_logs"
_CHECKPOINT_DIR = _PROJECT_ROOT / "models" / "checkpoints"

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS: int = 500_000
DEFAULT_N_ENVS: int = 4
DEFAULT_SEED: int = 42

PPO_HPARAMS: dict = {
    "n_steps": 2048,          # pasos por env antes de actualizar la política
    "batch_size": 64,         # tamaño del minibatch para el update
    "n_epochs": 10,           # épocas de optimización por update
    "gamma": 0.99,            # factor de descuento
    "gae_lambda": 0.95,       # lambda de GAE (trade-off bias/varianza)
    "clip_range": 0.2,        # ratio de clipping de PPO (ε)
    "ent_coef": 0.01,         # coeficiente de entropía (exploración)
    "vf_coef": 0.5,           # coeficiente de la función de valor
    "max_grad_norm": 0.5,     # clipping del gradiente
    "verbose": 1,
}


# ===========================================================================
# Utilidades
# ===========================================================================

def linear_schedule(initial_lr: float) -> Callable[[float], float]:
    """
    Genera un schedule de learning rate lineal decreciente.

    La tasa de aprendizaje comienza en ``initial_lr`` y decrece linealmente
    hasta 0 a lo largo del entrenamiento, siguiendo el progreso restante
    (``progress_remaining`` va de 1.0 a 0.0).

    Args:
        initial_lr: Tasa de aprendizaje inicial.

    Returns:
        Función ``fn(progress_remaining) → lr_actual``.
    """
    def fn(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return fn


def make_env_fn(seed: int = 0) -> Callable[[], TrafficEnv]:
    """
    Fábrica de entornos compatible con SubprocVecEnv.

    Cada proceso hijo recibe una semilla distinta para garantizar
    la selección independiente de escenarios de estrés en cada reset().

    Args:
        seed: Semilla base para este entorno.

    Returns:
        Función sin argumentos que retorna un TrafficEnv inicializado.
    """
    def _init() -> TrafficEnv:
        env = TrafficEnv(seed=seed)
        return env
    return _init


# ===========================================================================
# Callback de métricas personalizadas (TensorBoard)
# ===========================================================================

class TensorboardCallback(BaseCallback):
    """
    Callback que extrae métricas de equidad y eficiencia desde los ``info``
    dicts de los entornos vectorizados y las registra en TensorBoard.

    Métricas registradas en cada rollout:
        custom/total_wait   — Tiempo de espera total acumulado (s)
        custom/gini_index   — Índice de Gini de los tiempos de espera ∈ [0,1]
        custom/queue_total  — Longitud total de colas (vehículos detenidos)
        custom/mean_reward  — Recompensa media del paso actual

    La extracción se realiza sobre los infos del último paso de simulación,
    promediando sobre todos los entornos paralelos activos.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        # Buffers para acumular métricas entre dumps
        self._wait_buffer: list[float] = []
        self._gini_buffer: list[float] = []
        self._queue_buffer: list[float] = []

    def _on_step(self) -> bool:
        """
        Llamado tras cada step() del entorno vectorizado.

        Extrae los campos relevantes del diccionario ``infos`` (lista de dicts,
        uno por entorno paralelo) y los acumula en buffers internos.
        """
        infos: list[dict] = self.locals.get("infos", [])

        for info in infos:
            if not isinstance(info, dict):
                continue

            if "total_wait_s" in info:
                self._wait_buffer.append(float(info["total_wait_s"]))
            if "gini_index" in info:
                self._gini_buffer.append(float(info["gini_index"]))
            if "queue_total" in info:
                self._queue_buffer.append(float(info["queue_total"]))

        # Hacer dump a TensorBoard cada 512 pasos acumulados
        if len(self._wait_buffer) >= 512:
            self.logger.record(
                "custom/total_wait",
                float(np.mean(self._wait_buffer)),
            )
            self.logger.record(
                "custom/gini_index",
                float(np.mean(self._gini_buffer)),
            )
            self.logger.record(
                "custom/queue_total",
                float(np.mean(self._queue_buffer)),
            )
            # Log a consola cada dump
            if self.verbose >= 1:
                logger.info(
                    f"  [TB] step={self.num_timesteps:,} | "
                    f"wait={np.mean(self._wait_buffer):.1f}s | "
                    f"gini={np.mean(self._gini_buffer):.4f} | "
                    f"queue={np.mean(self._queue_buffer):.1f}"
                )
            # Limpiar buffers
            self._wait_buffer.clear()
            self._gini_buffer.clear()
            self._queue_buffer.clear()

        return True   # True = continuar entrenamiento


# ===========================================================================
# Pipeline de entrenamiento
# ===========================================================================

def train(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    seed: int = DEFAULT_SEED,
    model_name: str = "ppo_sumo_v1",
) -> PPO:
    """
    Ejecuta el pipeline completo de entrenamiento PPO:

        SubprocVecEnv (N entornos paralelos)
            → VecMonitor (logging automático de episodios)
            → PPO(MlpPolicy) con LR lineal
            → Callbacks: TensorBoard + Checkpoint + Eval
            → Guardado del modelo final

    Args:
        total_timesteps: Total de pasos de entorno para el entrenamiento.
        n_envs: Número de entornos paralelos (SubprocVecEnv).
        seed: Semilla global de reproducibilidad.
        model_name: Nombre del modelo para guardar en disk.

    Returns:
        Modelo PPO entrenado.
    """
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║      ENTRENAMIENTO PPO — Control Semafórico Equitativo   ║")
    logger.info("║      Framework TSC — Módulo 3: RL Agent                 ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Timesteps:   {total_timesteps:,}")
    logger.info(f"  Entornos:    {n_envs} (SubprocVecEnv)")
    logger.info(f"  Semilla:     {seed}")
    logger.info(f"  Modelo:      {model_name}")

    # ── Directorios de salida ─────────────────────────────────────────────
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    model_save_path = _MODELS_DIR / model_name
    eval_log_path = _MODELS_DIR / "eval_logs"
    eval_log_path.mkdir(parents=True, exist_ok=True)

    # ── PASO 1: Entornos vectorizados en paralelo ─────────────────────────
    logger.info("─" * 60)
    logger.info("PASO 1 — Instanciando entornos paralelos (SubprocVecEnv)")
    logger.info("─" * 60)

    # Cada entorno recibe una semilla distinta: seed, seed+1, ..., seed+N-1
    env_fns = [make_env_fn(seed=seed + i) for i in range(n_envs)]

    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(
        train_env,
        filename=str(_MODELS_DIR / "monitor_train"),
    )
    logger.info(f"  {n_envs} entornos de entrenamiento listos.")

    # Entorno de evaluación independiente (1 solo, no en subprocess)
    eval_env = make_vec_env(
        lambda: TrafficEnv(seed=seed + 999),
        n_envs=1,
    )
    eval_env = VecMonitor(eval_env, filename=str(eval_log_path / "monitor_eval"))
    logger.info("  1 entorno de evaluación listo.")

    # ── PASO 2: Definir callbacks ─────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("PASO 2 — Configurando callbacks")
    logger.info("─" * 60)

    # Guarda un checkpoint cada 50,000 pasos
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(_CHECKPOINT_DIR),
        name_prefix=model_name,
        verbose=1,
    )

    # Evalúa el agente cada 25,000 pasos y guarda el mejor modelo
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(_MODELS_DIR / "best_model"),
        log_path=str(eval_log_path),
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Callback personalizado para métricas de equidad en TensorBoard
    tb_cb = TensorboardCallback(verbose=1)

    callback_list = CallbackList([tb_cb, checkpoint_cb, eval_cb])
    logger.info("  Callbacks: TensorBoard + Checkpoint + Eval")

    # ── PASO 3: Instanciar el agente PPO ──────────────────────────────────
    logger.info("─" * 60)
    logger.info("PASO 3 — Instanciando agente PPO (MlpPolicy)")
    logger.info("─" * 60)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(3e-4),   # LR lineal decreciente
        **PPO_HPARAMS,
        seed=seed,
        tensorboard_log=str(_TB_LOG_DIR),
        device="cpu",                          # GPU si disponible, si no CPU
    )

    logger.info(f"  Política: MlpPolicy | Dispositivo: {model.device}")
    logger.info(f"  n_steps={PPO_HPARAMS['n_steps']} | "
                f"batch_size={PPO_HPARAMS['batch_size']} | "
                f"γ={PPO_HPARAMS['gamma']}")

    # ── PASO 4: Entrenamiento ─────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info(f"PASO 4 — Entrenando por {total_timesteps:,} pasos...")
    logger.info("─" * 60)
    logger.info("  Monitorea el progreso con:")
    logger.info(f"  tensorboard --logdir {_TB_LOG_DIR}")

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("  Entrenamiento interrumpido manualmente — guardando modelo parcial...")
    finally:
        # ── PASO 5: Guardar modelo final ──────────────────────────────────
        model.save(str(model_save_path))
        elapsed = time.time() - t0
        logger.info("─" * 60)
        logger.info(f"  Modelo guardado en: {model_save_path}.zip")
        logger.info(f"  Tiempo total de entrenamiento: {elapsed / 60:.1f} min")

        # Cerrar entornos
        train_env.close()
        eval_env.close()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║               ENTRENAMIENTO COMPLETADO                  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    return model


# ===========================================================================
# PUNTO DE ENTRADA
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena un agente PPO para control semafórico equitativo con SUMO."
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help=f"Total de pasos de entrenamiento (default: {DEFAULT_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--n-envs", type=int, default=DEFAULT_N_ENVS,
        help=f"Número de entornos en paralelo (default: {DEFAULT_N_ENVS})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Semilla de aleatoriedad (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--model-name", type=str, default="ppo_sumo_v1",
        help="Nombre del archivo de modelo guardado (default: ppo_sumo_v1)",
    )
    args = parser.parse_args()

    trained_model = train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        model_name=args.model_name,
    )

    logger.info(f"Modelo disponible en: models/{args.model_name}.zip")
    logger.info("Para evaluar: python src/rl_agent/evaluate.py --model models/ppo_sumo_v1")
