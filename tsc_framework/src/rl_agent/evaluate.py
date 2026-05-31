"""
evaluate.py — Evaluación visual del agente PPO entrenado
=========================================================
Carga el modelo PPO guardado, instancia TrafficEnv con SUMO-GUI activo
y ejecuta un episodio completo imprimiendo métricas de equidad en tiempo real.

Uso:
    python -m src.rl_agent.evaluate
    python -m src.rl_agent.evaluate --model models/ppo_sumo_v1 --deterministic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL = _PROJECT_ROOT / "models" / "ppo_sumo_v1"


# ===========================================================================
# Evaluación
# ===========================================================================

def evaluate(
    model_path: Path = _DEFAULT_MODEL,
    deterministic: bool = True,
    seed: int = 0,
) -> dict:
    """
    Ejecuta un episodio completo con el agente PPO entrenado en SUMO-GUI.

    Args:
        model_path: Ruta al modelo .zip (sin extensión).
        deterministic: Si True, usa política determinística (sin exploración).
        seed: Semilla para la selección del escenario de estrés.

    Returns:
        Diccionario con métricas del episodio:
            total_reward, total_wait_s, mean_gini, n_steps.
    """
    # ── Cargar modelo ─────────────────────────────────────────────────────
    zip_path = Path(str(model_path) + ".zip")
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {zip_path}\n"
            "Entrena primero con: python -m src.rl_agent.train"
        )

    logger.info(f"Cargando modelo: {zip_path}")
    model = PPO.load(str(model_path), device="cpu")
    logger.info("  ✓ Modelo cargado.")

    # ── Crear entorno con SUMO-GUI ─────────────────────────────────────────
    logger.info("Iniciando TrafficEnv con SUMO-GUI...")
    env = TrafficEnv(use_gui=True, seed=seed)

    # ── Episodio completo ─────────────────────────────────────────────────
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║            EVALUACIÓN — Episodio con SUMO-GUI            ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    obs, info = env.reset()
    logger.info(f"  Escenario: {info.get('scenario', 'N/A')}")
    logger.info(f"  Semáforo:  {info.get('tls_id', 'N/A')} | "
                f"Carriles: {info.get('n_controlled_lanes', '?')} | "
                f"Fases verdes: {info.get('n_green_phases', '?')}")
    logger.info("─" * 60)

    # Acumuladores de métricas
    total_reward: float = 0.0
    total_wait: float = 0.0
    gini_values: list[float] = []
    n_steps: int = 0

    try:
        terminated = truncated = False
        while not (terminated or truncated):
            # Acción del agente
            action, _ = model.predict(obs, deterministic=deterministic)

            # Ejecutar paso
            obs, reward, terminated, truncated, info = env.step(int(action))

            # Acumular métricas
            total_reward += float(reward)
            step_wait = info.get("total_wait_s", 0.0)
            step_gini = info.get("gini_index", 0.0)
            total_wait += step_wait
            gini_values.append(step_gini)
            n_steps += 1

            # Log cada 20 pasos
            if n_steps % 20 == 0:
                sim_time = info.get("sim_time_s", n_steps * env.delta_time)
                logger.info(
                    f"  t={sim_time:>7.1f}s | "
                    f"step={n_steps:>4} | "
                    f"reward={reward:>10.2f} | "
                    f"wait={step_wait:>8.1f}s | "
                    f"gini={step_gini:.4f} | "
                    f"fase={info.get('current_phase', '?')}"
                )

    except KeyboardInterrupt:
        logger.warning("  Evaluación interrumpida manualmente.")
    finally:
        env.close()

    # ── Resumen final ─────────────────────────────────────────────────────
    mean_gini = float(np.mean(gini_values)) if gini_values else 0.0
    max_gini  = float(np.max(gini_values))  if gini_values else 0.0

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║                 RESUMEN DEL EPISODIO                    ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Pasos ejecutados:       {n_steps}")
    logger.info(f"  Recompensa total:       {total_reward:.2f}")
    logger.info(f"  Espera total acumulada: {total_wait:.1f} s")
    logger.info(f"  Gini medio:             {mean_gini:.4f}  (0=equidad perfecta)")
    logger.info(f"  Gini máximo:            {max_gini:.4f}")
    logger.info(f"  Gini acumulado (Σ):     {sum(gini_values):.4f}")

    # Interpretación del Gini
    if mean_gini < 0.20:
        equity_label = "✅ Alta equidad"
    elif mean_gini < 0.40:
        equity_label = "⚠️  Equidad moderada"
    else:
        equity_label = "❌ Alta inequidad"
    logger.info(f"  Equidad distributiva:   {equity_label}")

    return {
        "total_reward": total_reward,
        "total_wait_s": total_wait,
        "mean_gini": mean_gini,
        "max_gini": max_gini,
        "cumulative_gini": sum(gini_values),
        "n_steps": n_steps,
    }


# ===========================================================================
# PUNTO DE ENTRADA
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalúa el agente PPO entrenado con SUMO-GUI."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(_DEFAULT_MODEL),
        help="Ruta al modelo (sin .zip). Default: models/ppo_sumo_v1",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Política determinística (sin exploración). Default: True",
    )
    parser.add_argument(
        "--stochastic",
        action="store_false",
        dest="deterministic",
        help="Política estocástica (con exploración).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semilla para selección de escenario de estrés. Default: 0",
    )
    args = parser.parse_args()

    metrics = evaluate(
        model_path=Path(args.model),
        deterministic=args.deterministic,
        seed=args.seed,
    )
