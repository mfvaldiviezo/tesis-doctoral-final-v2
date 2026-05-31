"""
train.py — Script Principal de Entrenamiento del Agente RL
===========================================================
Entrena un agente PPO sobre el entorno SumoRLEnv usando Stable-Baselines3.

Uso desde la raíz del proyecto:
    python scripts/train.py
    python scripts/train.py --config config/default_config.yaml
    python scripts/train.py --config config/default_config.yaml --n-envs 2
    python scripts/train.py --config config/default_config.yaml --no-vec

Salidas generadas:
    outputs/models/checkpoints/   — Checkpoints periódicos
    outputs/models/best_model.zip — Mejor política según evaluación
    outputs/models/ppo_final.zip  — Modelo final tras el entrenamiento
    outputs/tensorboard/          — Logs para TensorBoard
    outputs/logs/                 — Monitor CSV por episodio
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

# ── Asegurar que src/ esté en el PYTHONPATH ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# ── Imports del framework ─────────────────────────────────────────────────────
try:
    from src.core.tsc_env import TSCEnv, TSCConfig  # noqa: E402
except ImportError as e:
    raise ImportError(
        "TSCEnv no encontrado. Asegúrese de que src/core/tsc_env.py exista.\n"
        f"Error original: {e}"
    ) from e

# ── Imports de Stable-Baselines3 ─────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )
    from stable_baselines3.common.utils import set_random_seed
except ImportError as exc:
    raise ImportError(
        "Stable-Baselines3 no encontrado. "
        "Instala con: pip install stable-baselines3"
    ) from exc

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tsc.train")


# ─────────────────────────────────────────────────────────────────────────────
# Carga de configuración
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Carga el archivo YAML de configuración y retorna un diccionario anidado.

    Parameters
    ----------
    config_path : Path
        Ruta al archivo .yaml (absoluta o relativa al directorio de trabajo).

    Returns
    -------
    dict
        Configuración completa del framework.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {config_path}\n"
            f"Ejecuta el script desde la raíz del proyecto o usa --config <ruta>."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuración cargada desde: %s", config_path)
    return cfg


def _resolve_path(raw: str, root: Path) -> Path:
    """Convierte una ruta relativa del YAML en absoluta desde la raíz del proyecto."""
    p = Path(raw)
    return p if p.is_absolute() else root / p


# ─────────────────────────────────────────────────────────────────────────────
# Factory del entorno
# ─────────────────────────────────────────────────────────────────────────────

def make_env(
    network_file: str,
    route_file: str,
    tls_id: str,
    rank: int,
    seed: int = 42,
    use_gui: bool = False,
    step_length: int = 5,
    max_steps: int = 3600,
    device: str = "cpu",
    enable_traci_chaos: bool = False,
) -> Callable[[], Monitor]:
    """
    Función *factory* para SubprocVecEnv / DummyVecEnv.

    Cada proceso hijo llama a esta función para crear su propia instancia
    de TSCEnv + Monitor. Se asigna una semilla distinta a cada proceso
    mediante ``seed + rank`` para garantizar diversidad entre episodios.

    Parameters
    ----------
    network_file : str
        Ruta al archivo .net.xml de la red SUMO.
    route_file : str
        Ruta al archivo .rou.xml de rutas vehiculares.
    tls_id : str
        ID del semáforo a controlar (ej: "B1" para Hangzhou 4x4).
    rank : int
        Índice del entorno (0, 1, ..., n_envs-1).
    seed : int
        Semilla base; la semilla efectiva es seed + rank.
    use_gui : bool
        Si True lanza sumo-gui (solo recomendado para rank=0 y depuración).
    step_length : int
        Segundos SUMO por step de RL (delta_t).
    max_steps : int
        Pasos máximos por episodio.
    device : str
        Dispositivo para PyTorch: "cpu" (forzado) o "cuda".

    Returns
    -------
    Callable[[], Monitor]
        Thunk sin argumentos que construye y retorna el entorno envuelto.
    """
    def _thunk() -> Monitor:
        set_random_seed(seed + rank)
        
        # Construir ruta al archivo .sumocfg temporal requerido por TSCEnv
        # TSCEnv usa sumocfg_path como punto de entrada principal
        # Generamos un sumocfg dinámico apuntando a network_file y route_file
        import tempfile
        import os
        
        # Crear un archivo .sumocfg temporal que referencie la red y las rutas
        tmp_dir = tempfile.gettempdir()
        sumocfg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{network_file}"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{max_steps * step_length}"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <waiting-time-memory value="1000"/>
    </processing>
    <random_number>
        <random value="true"/>
    </random_number>
</configuration>
'''
        sumocfg_path = os.path.join(tmp_dir, f"tsc_env_rank{rank}_seed{seed}.sumocfg")
        with open(sumocfg_path, 'w', encoding='utf-8') as f:
            f.write(sumocfg_content)
        
        # Configurar parámetros del entorno (TSCEnv signature: sumocfg_path, tls_id, use_gui, delta_t, max_steps, ...)
        env_config = {
            "sumocfg_path": sumocfg_path,
            "tls_id": tls_id,
            "delta_t": step_length,       # TSCEnv usa delta_t, no step_length
            "max_steps": max_steps,
            "use_gui": use_gui and rank == 0,  # GUI solo en proceso 0
            "seed": seed + rank,
            "enable_traci_chaos": enable_traci_chaos,
        }
        
        env = TSCEnv(**env_config)
        return Monitor(env)

    return _thunk


# ─────────────────────────────────────────────────────────────────────────────
# Callback personalizado: Métricas de riesgo en TensorBoard
# ─────────────────────────────────────────────────────────────────────────────

class RiskMetricsCallback(BaseCallback):
    """
    Callback que extrae las métricas de riesgo del diccionario ``info`` de
    cada step y las registra en TensorBoard.

    Métricas registradas:
        custom/gini_mean    — Media del Índice de Gini sobre los entornos activos
        custom/cvar_mean    — Media del CVaR₉₀ sobre los entornos activos
        custom/delay_mean   — Media del delay agregado
        custom/total_queue  — Suma total de vehículos detenidos

    El diccionario ``info`` es producido por SumoRLEnv.step() en cada
    transición. En entornos vectorizados, ``self.locals["infos"]`` es una
    lista de dicts (uno por subproceso).
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._gini_buffer:  List[float] = []
        self._cvar_buffer:  List[float] = []
        self._delay_buffer: List[float] = []
        self._queue_buffer: List[float] = []

    def _on_step(self) -> bool:
        """
        Llamado en cada step del entorno vectorizado.
        Retorna True para continuar el entrenamiento.
        """
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])

        for info in infos:
            # SumoRLEnv puede poner las métricas en info directamente.
            # Si no están (ej. episodio truncado), se ignora ese paso.
            if "gini" in info:
                self._gini_buffer.append(float(info["gini"]))
            if "cvar" in info:
                self._cvar_buffer.append(float(info["cvar"]))
            if "delay" in info:
                self._delay_buffer.append(float(info["delay"]))
            if "total_queue" in info:
                self._queue_buffer.append(float(info["total_queue"]))

        # Registrar en TensorBoard cada 100 steps
        if self.n_calls % 100 == 0:
            self._flush_to_tensorboard()

        return True  # False detendría el entrenamiento

    def _flush_to_tensorboard(self) -> None:
        """Vuelca los buffers acumulados a TensorBoard y los vacía."""
        if self._gini_buffer:
            self.logger.record("custom/gini_mean",  np.mean(self._gini_buffer))
            self._gini_buffer.clear()
        if self._cvar_buffer:
            self.logger.record("custom/cvar_mean",  np.mean(self._cvar_buffer))
            self._cvar_buffer.clear()
        if self._delay_buffer:
            self.logger.record("custom/delay_mean", np.mean(self._delay_buffer))
            self._delay_buffer.clear()
        if self._queue_buffer:
            self.logger.record("custom/total_queue", np.mean(self._queue_buffer))
            self._queue_buffer.clear()

    def _on_training_end(self) -> None:
        """Vuelca cualquier dato restante al finalizar el entrenamiento."""
        self._flush_to_tensorboard()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSC Framework — Entrenamiento PPO sobre SumoRLEnv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default_config.yaml",
        help="Ruta al archivo de configuración YAML.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Número de entornos paralelos (sobreescribe config YAML).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total de timesteps de entrenamiento (sobreescribe config YAML).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["PPO"],          # SAC/TD3 se añadirán en Fase siguiente
        default="PPO",
        help="Algoritmo RL a usar.",
    )
    parser.add_argument(
        "--no-vec",
        action="store_true",
        help="Usar DummyVecEnv en lugar de SubprocVecEnv (útil para depurar).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Lanzar SUMO con interfaz gráfica (solo para depuración).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla global (sobreescribe config YAML).",
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="Habilitar el comportamiento de tráfico caótico/imprudente LATAM en el entrenamiento.",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["hsarg", "coop"],
        default="hsarg",
        help="Extractor de características RL a utilizar: H-SARG (hsarg) o Coop-SARG (coop).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Cargar configuración ──────────────────────────────────────────────
    cfg = load_config(args.config)

    repro_cfg  = cfg.get("reproducibility", {})
    sumo_cfg   = cfg.get("sumo",            {})
    agent_cfg  = cfg.get("agent",           {})
    ppo_cfg    = agent_cfg.get("ppo",       {})
    policy_cfg = agent_cfg.get("policy",    {})
    paths_cfg  = cfg.get("paths",           {})
    env_cfg    = cfg.get("environment",     {})
    log_cfg    = cfg.get("logging",         {})
    risk_cfg   = cfg.get("risk_metrics",    {})

    # Configurar SUMO_HOME desde YAML o variable de entorno (Windows/Linux)
    _sumo_home = sumo_cfg.get("sumo_home", "")
    if _sumo_home:
        os.environ["SUMO_HOME"] = _sumo_home
    elif not os.environ.get("SUMO_HOME"):
        logger.warning(
            "SUMO_HOME no está configurado. Asegúrate de instalar SUMO y configurar "
            "la variable de entorno o establecer 'sumo_home' en el YAML."
        )

    # Resolución de valores con prioridad: CLI > YAML > default
    SEED          = args.seed       or repro_cfg.get("global_seed", 42)
    N_ENVS        = args.n_envs     or agent_cfg.get("n_envs",       4)
    TOTAL_TS      = args.timesteps  or agent_cfg.get("total_timesteps", 1_000_000)
    LR            = float(agent_cfg.get("learning_rate", 3e-4))
    BATCH_SIZE    = int(agent_cfg.get("batch_size",     64))
    N_STEPS       = int(ppo_cfg.get("n_steps",          2048))
    N_EPOCHS      = int(ppo_cfg.get("n_epochs",         10))
    CLIP_RANGE    = float(ppo_cfg.get("clip_range",     0.2))
    ENT_COEF      = float(ppo_cfg.get("ent_coef",       0.01))
    GAE_LAMBDA    = float(ppo_cfg.get("gae_lambda",     0.95))
    GAMMA         = float(agent_cfg.get("gamma",        0.99))
    DEVICE        = agent_cfg.get("device",             "auto")
    NET_ARCH      = policy_cfg.get("net_arch",          [256, 256])
    SAVE_FREQ     = int(log_cfg.get("save_freq",        50_000))
    EVAL_FREQ     = int(cfg.get("evaluation", {}).get("eval_freq", 10_000))
    N_EVAL_EPS    = int(cfg.get("evaluation", {}).get("n_eval_episodes", 10))
    DELTA_T       = int(sumo_cfg.get("step_length",     5))
    MAX_STEPS     = int(sumo_cfg.get("end_time",        3600))

    REWARD_WEIGHTS = {
        "delay":    float(env_cfg.get("reward_weights", {}).get("queue_length", 0.4)),
        "pressure": float(env_cfg.get("reward_weights", {}).get("avg_speed",    0.3)),
        "gini":     float(risk_cfg.get("gini_weight",                           0.15)),
        "cvar":     float(risk_cfg.get("cvar_alpha",                            0.15)),
    }

    # Normalizar pesos para que sumen 1
    _total_w = sum(REWARD_WEIGHTS.values())
    if abs(_total_w - 1.0) > 1e-3:
        logger.warning("Los pesos de recompensa no suman 1 (Σ=%.3f). Normalizando.", _total_w)
        REWARD_WEIGHTS = {k: v / _total_w for k, v in REWARD_WEIGHTS.items()}

    # ── 2. Rutas de salida ───────────────────────────────────────────────────
    # Las rutas reales se extraen de la sección 'network' (benchmark o custom)
    TB_LOG_DIR    = PROJECT_ROOT / paths_cfg.get("tensorboard_dir", "outputs/tensorboard")
    MODELS_DIR    = PROJECT_ROOT / paths_cfg.get("models_dir",      "outputs/models")
    CKPT_DIR      = MODELS_DIR / "checkpoints"
    BEST_DIR      = MODELS_DIR / "best_model"
    MONITOR_DIR   = PROJECT_ROOT / paths_cfg.get("logs_dir",        "outputs/logs")

    for d in (TB_LOG_DIR, CKPT_DIR, BEST_DIR, MONITOR_DIR):
        d.mkdir(parents=True, exist_ok=True)

    set_random_seed(SEED)

    logger.info("=" * 60)
    logger.info("  TSC Framework — Entrenamiento PPO")
    logger.info("  Timesteps   : %s", f"{TOTAL_TS:,}")
    logger.info("  N Envs      : %d", N_ENVS)
    logger.info("  Semilla     : %d", SEED)
    logger.info("  Device      : %s", DEVICE)
    logger.info("=" * 60)

    # ── 4. Construir entornos vectorizados ───────────────────────────────────
    # Extraer rutas de red y rutas desde la configuración (Cap. 4.2.1 - Estado 34D)
    network_mode = cfg.get("network", {}).get("mode", "benchmark")
    if network_mode == "benchmark":
        net_cfg = cfg.get("network", {}).get("benchmark", {})
        NETWORK_FILE = _resolve_path(net_cfg.get("network_file", ""), PROJECT_ROOT)
        ROUTE_FILES_DIR = _resolve_path(net_cfg.get("route_files_dir", ""), PROJECT_ROOT)
        TLS_ID = net_cfg.get("tls_id", "B1")
    else:
        custom_cfg = cfg.get("network", {}).get("custom", {})
        NETWORK_FILE = _resolve_path(custom_cfg.get("net_output", ""), PROJECT_ROOT)
        ROUTE_FILES_DIR = _resolve_path(custom_cfg.get("route_file", ""), PROJECT_ROOT)
        TLS_ID = "J0"
    
    # Obtener primer archivo de ruta para el entorno
    route_dir = Path(ROUTE_FILES_DIR) if ROUTE_FILES_DIR else None
    if route_dir and route_dir.exists():
        route_files = sorted(route_dir.glob("*.rou.xml"))
        ROUTE_FILE = route_files[0] if route_files else route_dir / "hangzhou_minimal.rou.xml"
    else:
        ROUTE_FILE = PROJECT_ROOT / "sumo_configs/routes/hangzhou/hangzhou_minimal.rou.xml"
    
    env_factories: List[Callable] = [
        make_env(
            network_file=str(NETWORK_FILE),
            route_file=str(ROUTE_FILE),
            tls_id=TLS_ID,
            rank=i,
            seed=SEED,
            use_gui=args.gui,
            step_length=DELTA_T,
            max_steps=MAX_STEPS,
            device="cpu",  # CPU-forzado explícito (Cap. 4.3)
            enable_traci_chaos=args.chaos,
        )
        for i in range(N_ENVS)
    ]

    VecEnvClass = DummyVecEnv if args.no_vec or N_ENVS == 1 else SubprocVecEnv
    logger.info(
        "Construyendo %d entornos con %s...",
        N_ENVS,
        VecEnvClass.__name__,
    )
    vec_env = VecEnvClass(env_factories)
    vec_env = VecMonitor(vec_env, filename=str(MONITOR_DIR / "monitor"))
    logger.info("Entorno vectorizado listo.")

    # ── 5. Entorno de evaluación (síncrono, sin GUI) ─────────────────────────
    eval_env = DummyVecEnv(
        [
            make_env(
                network_file=str(NETWORK_FILE),
                route_file=str(ROUTE_FILE),
                tls_id=TLS_ID,
                rank=N_ENVS,       # rank distinto para semilla independiente
                seed=SEED + 1000,  # semilla completamente separada
                use_gui=False,
                step_length=DELTA_T,
                max_steps=MAX_STEPS,
                device="cpu",      # CPU-forzado explícito (Cap. 4.3)
                enable_traci_chaos=args.chaos,
            )
        ]
    )
    eval_env = VecMonitor(eval_env, filename=str(MONITOR_DIR / "eval_monitor"))

    # ── 6. Instanciar PPO con el extractor seleccionado ──────────────────────
    features_extractor = None
    if args.extractor == "coop":
        try:
            from src.rl_agent.sarg_policy import CoopSARGExtractor
            features_extractor = CoopSARGExtractor
            logger.info("⚡ Utilizando el nuevo extractor Coop-SARG (Cooperative Self-Attention Gated Risk).")
        except ImportError:
            try:
                from tsc_framework.src.rl_agent.sarg_policy import CoopSARGExtractor
                features_extractor = CoopSARGExtractor
                logger.info("⚡ Utilizando el nuevo extractor Coop-SARG (Cooperative Self-Attention Gated Risk).")
            except ImportError as e:
                logger.warning("No se pudo cargar CoopSARGExtractor. Error: %s", e)
    else:
        try:
            from src.rl_agent.sarg_policy import HSARGExtractor
            features_extractor = HSARGExtractor
            logger.info("🧠 Utilizando el extractor H-SARG nominal.")
        except ImportError:
            try:
                from tsc_framework.src.rl_agent.sarg_policy import HSARGExtractor
                features_extractor = HSARGExtractor
                logger.info("🧠 Utilizando el extractor H-SARG nominal.")
            except ImportError as e:
                logger.warning("No se pudo cargar HSARGExtractor. Error: %s", e)

    policy_kwargs: Dict[str, Any] = {}
    if features_extractor is not None:
        policy_kwargs = {
            "features_extractor_class": features_extractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": dict(pi=NET_ARCH, vf=NET_ARCH),
        }
    else:
        policy_kwargs = {
            "net_arch": NET_ARCH,
        }


    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(TB_LOG_DIR),
        verbose=1,
        seed=SEED,
        device=DEVICE,
    )

    logger.info("Modelo PPO instanciado.")
    logger.info("  Policy net  : %s", NET_ARCH)
    logger.info("  LR          : %s", LR)
    logger.info("  n_steps     : %d | batch_size: %d | n_epochs: %d",
                N_STEPS, BATCH_SIZE, N_EPOCHS)

    # ── 7. Callbacks ─────────────────────────────────────────────────────────

    # 7a. Checkpoint periódico
    checkpoint_cb = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1),   # ajustar a pasos por env
        save_path=str(CKPT_DIR),
        name_prefix="ppo_tsc",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 7b. Evaluación y guardado del mejor modelo
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(MONITOR_DIR / "eval"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPS,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 7c. Métricas de riesgo personalizadas en TensorBoard
    risk_cb = RiskMetricsCallback(verbose=0)

    callback_list = CallbackList([checkpoint_cb, eval_cb, risk_cb])

    # ── 8. Entrenamiento ─────────────────────────────────────────────────────
    logger.info("Iniciando entrenamiento | total_timesteps=%s", f"{TOTAL_TS:,}")
    try:
        model.learn(
            total_timesteps=TOTAL_TS,
            callback=callback_list,
            tb_log_name="PPO_TSC",
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("Entrenamiento interrumpido por el usuario (Ctrl+C).")
    except Exception as exc:
        logger.exception("Error durante el entrenamiento: %s", exc)
        raise

    # ── 9. Guardar modelo final ───────────────────────────────────────────────
    final_path = MODELS_DIR / "ppo_final"
    model.save(str(final_path))
    logger.info("✅ Modelo final guardado en: %s.zip", final_path)

    # ── 10. Cierre de entornos ────────────────────────────────────────────────
    vec_env.close()
    eval_env.close()
    logger.info("Entornos cerrados correctamente.")
    logger.info("Para ver TensorBoard ejecuta:")
    logger.info("    tensorboard --logdir %s", TB_LOG_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
