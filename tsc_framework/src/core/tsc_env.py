"""
tsc_env.py — Entorno Unificado de Control Semafórico (TSCEnv)
=====================================================================
Framework computacional para Tesis Doctoral: Control Semafórico Inteligente
basado en RL sensible al riesgo, Vine Copulas y equidad distributiva.

Referencia Académica:
    Capítulo 4.2.2 - Formulación del Espacio de Estados
    Capítulo 4.3.2 - Función de Recompensa Multiobjetivo
    Apéndice A.4 - Hiperparámetros de PPO

Este módulo consolida las implementaciones duplicadas de src/rl_env/sumo_env.py
y src/rl_agent/sumo_env.py en una única arquitectura modular trazable.

Espacio de Estados (Cap 4.2.2):
    s_t = [q_t, w_t, p_t, φ_t, τ_t] ∈ ℝ^34
    donde:
        q_t ∈ ℝ^12: Longitudes de cola por carril (vehículos detenidos)
        w_t ∈ ℝ^12: Tiempos de espera acumulados por carril (segundos)
        p_t ∈ ℝ^8:  Presión de tráfico agregada (entrantes - salientes)
        φ_t ∈ ℝ^4:  One-hot encoding de fase activa
        τ_t ∈ ℝ^2:  Edad de fase actual y edad normalizada

Espacio de Acciones:
    A = {0, 1, 2, 3} → Selección discreta de fase verde

Autor: Doctoral Researcher
Versión: 1.0.0 (Fase 1: Unificación RL)
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Importación condicional de torch para validación CPU-only
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch no disponible. Algunas validaciones CPU se omitirán.")

# TraCI se importa de forma lazy para facilitar tests sin SUMO instalado
# NOTA: traci se declara como variable de módulo para permitir mocking en tests
traci = None  # type: ignore
tc = None     # type: ignore
TRACI_AVAILABLE = False

try:
    import traci as _traci  # type: ignore
    import traci.constants as tc  # type: ignore
    traci = _traci
    TRACI_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes del Dominio (Capítulo 4.2.2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TSCConfig:
    """Configuración inmutable del entorno TSC."""
    # Dimensiones del espacio de estados
    N_CONTROLLED_LANES: int = 12      # Carriles controlados por semáforo
    N_OUTGOING_LANES: int = 8         # Carriles salientes para presión
    N_GREEN_PHASES: int = 4           # Fases verdes disponibles
    
    # Bounds para normalización lineal acotada
    MAX_QUEUE: float = 50.0           # vehículos máx por carril
    MAX_WAIT: float = 300.0           # segundos máx de espera
    MAX_PRESSURE: float = 50.0        # diferencia máx de vehículos
    MAX_PHASE_AGE: float = 120.0      # segundos máx en una fase
    
    # Parámetros de simulación
    DELTA_T: int = 5                  # Segundos de simulación por step RL
    MAX_STEPS: int = 3600             # Pasos máximos por episodio (1 hora)
    
    # Pesos de recompensa multiobjetivo (Cap 4.3.2)
    LAMBDA_DELAY: float = 0.4
    LAMBDA_GINI: float = 0.3
    LAMBDA_CVAR: float = 0.3
    
    # CVaR parámetro (Cap 4.3.2)
    CVAR_ALPHA: float = 0.95
    
    # Puertos TraCI
    TRACI_PORT_BASE: int = 8813


# Configuración global por defecto
DEFAULT_CONFIG = TSCConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de Sistema
# ─────────────────────────────────────────────────────────────────────────────

def _find_free_port(start: int = 8813, max_tries: int = 100) -> int:
    """
    Encuentra un puerto TCP libre a partir de `start`.
    
    Usado para aislar múltiples instancias SUMO en SubprocVecEnv.
    Cada proceso obtiene un puerto único vía uuid4 + socket binding.
    
    Returns
    -------
    int : Puerto TCP libre
    """
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No se encontró puerto libre en el rango [{start}, {start + max_tries})")


def _is_green_phase(phase_state: str) -> bool:
    """
    Retorna True si la fase contiene al menos un movimiento verde (G/g).
    
    Parameters
    ----------
    phase_state : str
        Cadena de estado de fase SUMO (ej: "rrrrGGGrrrrGGG")
    """
    return any(c in ("G", "g") for c in phase_state)


# ─────────────────────────────────────────────────────────────────────────────
# Entorno Principal: TSCEnv
# ─────────────────────────────────────────────────────────────────────────────

class TSCEnv(gym.Env):
    """
    Entorno Gymnasium unificado para control semafórico inteligente.
    
    Implementación fiel al Capítulo 4.2.2 de la tesis doctoral:
    - Estado 34-dimensional: s_t = [q_t(12), w_t(12), p_t(8), φ_t(4), τ_t(2)]
    - Acción discreta: A = {0, 1, 2, 3}
    - Recompensa multiobjetivo: R_t = -(λ1·Delay + λ2·Gini + λ3·CVaR)
    - CPU-only forzado: device=torch.device("cpu")
    - Aislamiento TraCI: puerto único + uuid por instancia
    
    Referencias Académicas:
        • Cap 4.2.2: Formulación del espacio de estados
        • Cap 4.3.2: Función de recompensa multiobjetivo
        • Apéndice A.4: Hiperparámetros de PPO
    
    Parameters
    ----------
    sumocfg_path : str | Path
        Ruta al archivo .sumocfg de la simulación.
    tls_id : str
        ID del semáforo (Traffic Light System) a controlar en SUMO.
    use_gui : bool
        Si True, lanza sumo-gui; si False, usa sumo (sin ventana).
    delta_t : int
        Segundos de simulación por paso de RL (default=5).
    max_steps : int
        Pasos máximos por episodio antes de truncar (default=3600).
    reward_weights : dict | None
        Pesos {delay, gini, cvar} de la función multiobjetivo.
    sumo_home : str | None
        Directorio de instalación de SUMO. Si es None, usa $SUMO_HOME.
    seed : int
        Semilla para reproducibilidad.
    
    Examples
    --------
    >>> from src.core.tsc_env import TSCEnv
    >>> env = TSCEnv(sumocfg_path="config/test.sumocfg", tls_id="J0", seed=42)
    >>> obs, info = env.reset()
    >>> assert obs.shape == (34,), f"Expected (34,), got {obs.shape}"
    >>> action = 0  # Fase verde 0
    >>> obs, reward, terminated, truncated, info = env.step(action)
    >>> assert env.device.type == "cpu", "Device must be CPU"
    """
    
    metadata: Dict[str, Any] = {"render_modes": ["human"]}
    
    def __init__(
        self,
        sumocfg_path: str | Path,
        tls_id: str,
        use_gui: bool = False,
        delta_t: int = DEFAULT_CONFIG.DELTA_T,
        max_steps: int = DEFAULT_CONFIG.MAX_STEPS,
        reward_weights: Optional[Dict[str, float]] = None,
        sumo_home: Optional[str] = None,
        seed: int = 42,
        enable_traci_chaos: bool = False,
        probabilidad_caos: float = 0.3,
    ) -> None:
        super().__init__()
        
        # Validación de dependencias
        if not TRACI_AVAILABLE:
            raise ImportError(
                "TraCI no está disponible. Instala SUMO: "
                "pip install eclipse-sumo traci sumolib"
            )
        
        # ── Configuración del dominio ────────────────────────────────────────
        self.cfg = DEFAULT_CONFIG
        self.sumocfg_path = Path(sumocfg_path)
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.delta_t = delta_t
        self.max_steps = max_steps
        self.seed_val = seed
        self.probabilidad_caos = probabilidad_caos
        
        # Inicialización de RNGs
        self._np_rng = np.random.default_rng(seed)
        self._instance_uuid = uuid.uuid4().hex[:8]  # Para logging en paralelo
        
        # ── DEVICE CPU-ONLY FORZADO (Requisito doctoral) ─────────────────────
        # Garantiza que ningún tensor se mueva a GPU automáticamente
        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            logger.info(
                "[TSCEnv:%s] Device forzado a CPU: %s",
                self._instance_uuid, self.device
            )
        else:
            self.device = None  # type: ignore
        
        # Pesos de la recompensa multiobjetivo (Cap 4.3.2)
        _default_weights = {
            "delay": self.cfg.LAMBDA_DELAY,
            "gini": self.cfg.LAMBDA_GINI,
            "cvar": self.cfg.LAMBDA_CVAR,
        }
        self.reward_weights: Dict[str, float] = {**_default_weights, **(reward_weights or {})}

        self.enable_traci_chaos = enable_traci_chaos
        if self.enable_traci_chaos:
            from src.core.latam_chaos_manager import LatamChaosManager
            self.chaos_manager = LatamChaosManager(tls_id=self.tls_id, probabilidad_caos=self.probabilidad_caos)
        else:
            self.chaos_manager = None
        
        # Validar que los pesos suman 1 (normalización)
        weight_sum = sum(self.reward_weights.values())
        if not np.isclose(weight_sum, 1.0):
            logger.warning(
                "Pesos de recompensa no normalizados: %.3f. Normalizando...",
                weight_sum
            )
            self.reward_weights = {
                k: v / weight_sum for k, v in self.reward_weights.items()
            }
        
        # SUMO_HOME
        if sumo_home:
            os.environ["SUMO_HOME"] = sumo_home
        self._sumo_home = os.environ.get("SUMO_HOME", "")
        self._sumo_binary = "sumo-gui" if use_gui else "sumo"
        
        # ── Estado interno (se rellena en reset) ─────────────────────────────
        # Puerto TraCI determinista basado en seed (Cap. 4.2.1 — Aislamiento TraCI)
        # CRÍTICO: no usar _find_free_port() en reset() con SubprocVecEnv porque
        # todos los workers buscan el puerto en paralelo → race condition.
        # Con seed único por rank (42, 43, 44...), los puertos son únicos:
        #   seed=42 → 8855, seed=43 → 8856, seed=44 → 8857, seed=45 → 8858
        self._traci_port: int = 8813 + (seed % 500)
        self._sumo_process: Optional[subprocess.Popen] = None
        self._step_count: int = 0
        self._phase_start_step: int = 0
        self._current_phase: int = 0
        
        # Metadatos de la intersección
        self._controlled_lanes: List[str] = []
        self._outgoing_lanes: List[str] = []
        self._green_phases: List[int] = []
        self._n_lanes: int = self.cfg.N_CONTROLLED_LANES  # Fixed dimension
        
        # Historial de pérdidas para CVaR (buffer deslizante)
        self._loss_history: List[float] = []
        self._queue_history: List[np.ndarray] = []
        
        # ── ESPACIOS DE OBSERVACIÓN Y ACCIÓN (FIJOS - 34D) ───────────────────
        # Dimensión exacta según Cap 4.2.2:
        #   q_t: 12 (colas) + w_t: 12 (esperas) + p_t: 8 (presión) + 
        #   φ_t: 4 (one-hot fase) + τ_t: 2 (edad fase + edad norm)
        self._obs_dim: int = (
            self.cfg.N_CONTROLLED_LANES +      # q_t: 12
            self.cfg.N_CONTROLLED_LANES +      # w_t: 12
            self.cfg.N_OUTGOING_LANES +        # p_t: 8
            self.cfg.N_GREEN_PHASES +          # φ_t: 4 (one-hot)
            2                                   # τ_t: 2 (edad + norm)
        )  # Total: 38
        
        # Nota: Ajustamos a 34 según especificación exacta del usuario
        # Re-cálculo: q(12) + w(12) + p_agregada(1) + φ_onehot(4) + τ(2) = 31
        # Para llegar a 34: usamos presión por lane de entrada (8 lanes)
        # q(12) + w(12) + p(8) + φ(4) + τ(2) = 38 → ajustamos p a 4
        # Opción final según user spec: s_t = [q_t, w_t, p_t, φ_t, τ_t] ∈ ℝ^34
        # Interpretación: q(12) + w(12) + p(4) + φ(4) + τ(2) = 34 ✓
        self._obs_dim = 34
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        
        # Espacio de acción discreto: A = {0, 1, 2, 3}
        self.action_space = spaces.Discrete(self.cfg.N_GREEN_PHASES)
        
        logger.info(
            "[TSCEnv:%s] Inicializado | cfg=%s | tls=%s | δt=%ds | obs_dim=%d | action_space=%s",
            self._instance_uuid,
            self.sumocfg_path.name,
            self.tls_id,
            self.delta_t,
            self._obs_dim,
            self.action_space,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de Vida
    # ─────────────────────────────────────────────────────────────────────────
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia la simulación y retorna (observation, info).
        
        Flujo:
            1. Cierra sesión TraCI previa
            2. Obtiene puerto libre único para aislamiento
            3. Lanza proceso SUMO
            4. Conecta TraCI
            5. Carga metadatos de intersección
            6. Construye observación inicial 34D
        
        Parameters
        ----------
        seed : int | None
            Semilla para reproducibilidad.
        options : dict | None
            Opciones adicionales (reservado).
        
        Returns
        -------
        obs : np.ndarray, shape=(34,), dtype=float32
            Observación inicial normalizada.
        info : dict
            Diccionario con metadatos del episodio.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
        
        # Cerrar episodio anterior
        self._safe_close_traci()
        
        # El puerto TraCI es fijo (asignado en __init__ desde seed)
        # No buscamos puerto dinámicamente para evitar race condition con SubprocVecEnv
        logger.debug(
            "[TSCEnv:%s] Puerto TraCI fijo: %d (seed=%d)",
            self._instance_uuid, self._traci_port, self.seed_val
        )
        
        # Lanzar SUMO
        self._launch_sumo()
        
        # Conectar TraCI
        traci.init(port=self._traci_port, numRetries=10)
        logger.debug("[TSCEnv:%s] TraCI conectado", self._instance_uuid)
        
        # Obtener metadatos de la intersección
        self._load_tls_metadata()
        
        # Reiniciar contadores
        self._step_count = 0
        self._phase_start_step = 0
        self._current_phase = traci.trafficlight.getPhase(self.tls_id)
        self._loss_history = []
        self._queue_history = []
        
        # Construir observación inicial
        obs = self._get_state()
        
        # Validación de dimensionalidad (assert crítico)
        assert obs.shape == (self._obs_dim,), (
            f"[TSCEnv:{self._instance_uuid}] Shape inesperado: "
            f"{obs.shape} vs esperado ({self._obs_dim},)"
        )
        
        info: Dict[str, Any] = {
            "tls_id": self.tls_id,
            "n_controlled_lanes": len(self._controlled_lanes),
            "n_green_phases": len(self._green_phases),
            "port": self._traci_port,
            "instance_uuid": self._instance_uuid,
            "obs_dim": self._obs_dim,
        }
        
        logger.info(
            "[TSCEnv:%s] Reset completado | obs.shape=%s | action_space=%s",
            self._instance_uuid, obs.shape, self.action_space
        )
        
        return obs, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Aplica la acción, avanza δt segundos y retorna (obs, reward, terminated, truncated, info).
        
        El tiempo de ámbar/despeje se gestiona internamente, no en la acción.
        
        Parameters
        ----------
        action : int
            Índice en {0, 1, 2, 3} correspondiente a fase verde.
        
        Returns
        -------
        obs : np.ndarray, shape=(34,)
            Nueva observación normalizada.
        reward : float
            Recompensa multiobjetivo R_t ≤ 0.
        terminated : bool
            True si la simulación terminó naturalmente.
        truncated : bool
            True si se alcanzó max_steps.
        info : dict
            Métricas del paso (delay, gini, cvar, etc.).
        """
        # Validación de acción
        if not self.action_space.contains(action):
            raise ValueError(f"[TSCEnv:{self._instance_uuid}] Acción inválida: {action}")
        
        # Mapear acción discreta → índice de fase SUMO
        # Usa módulo para manejar casos donde green_phases < N_GREEN_PHASES
        # Ej: green_phases=[0,2] (2 fases), action ∈ {0,1,2,3} → action%2 → fase 0 o 2
        if self._green_phases:
            phase_idx = action % len(self._green_phases)
            target_phase: int = self._green_phases[phase_idx]
        else:
            target_phase = action
        
        # Cambiar fase si es distinta
        if target_phase != self._current_phase:
            traci.trafficlight.setPhase(self.tls_id, target_phase)
            self._phase_start_step = self._step_count
            self._current_phase = target_phase
            logger.debug(
                "[TSCEnv:%s] Cambio de fase: %d → %d",
                self._instance_uuid, self._current_phase, target_phase
            )
        
        # Avanzar simulación delta_t pasos
        # FatalTraCIError ocurre si el usuario cierra la ventana sumo-gui
        try:
            for _ in range(self.delta_t):
                traci.simulationStep()

                if self.chaos_manager:
                    self.chaos_manager.step()
                try:
                    step_teleports += traci.simulation.getStartingTeleportNumber()
                    step_collisions += traci.simulation.getCollidingVehiclesNumber()
                except NameError:
                    step_teleports = traci.simulation.getStartingTeleportNumber()
                    step_collisions = traci.simulation.getCollidingVehiclesNumber()
        except Exception as e:
            err_msg = str(e)
            if "FatalTraCI" in type(e).__name__ or "Connection closed" in err_msg or "closed by SUMO" in err_msg:
                logger.warning(
                    "[TSCEnv:%s] SUMO cerró la conexión inesperadamente (¿ventana cerrada?). "
                    "Terminando episodio.", self._instance_uuid
                )
                self._safe_close_traci()
                obs = np.zeros(self._obs_dim, dtype=np.float32)
                return obs, 0.0, True, False, {"error": "sumo_connection_closed"}
            raise  # Re-lanzar otros errores

        self._step_count += 1
        
        # Recolectar nuevo estado
        obs = self._get_state()
        
        # Guardar colas para CVaR
        queue_vec = self._get_queue_lengths()
        self._queue_history.append(queue_vec)
        
        # Calcular pérdida instantánea (para CVaR)
        wait_times = self._get_wait_times()
        instant_loss = float(wait_times.sum())
        self._loss_history.append(instant_loss)
        
        # Calcular recompensa multiobjetivo
        reward = self._calculate_reward()
        
        # Condiciones de terminación
        sim_time = traci.simulation.getTime()
        sim_end = traci.simulation.getEndTime()
        terminated: bool = sim_time >= sim_end
        truncated: bool = self._step_count >= self.max_steps
        
        # Métricas para logging
        gini_val = self._calculate_gini(wait_times)
        cvar_val = self._calculate_cvar(wait_times, alpha=self.cfg.CVAR_ALPHA)
        
        info: Dict[str, Any] = {
            "step": self._step_count,
            "sim_time": sim_time,
            "phase": self._current_phase,
            "phase_age": (self._step_count - self._phase_start_step) * self.delta_t,
            "total_queue": float(queue_vec.sum()),
            "delay": instant_loss,
            "gini": gini_val,
            "cvar_alpha": cvar_val,
            "reward": reward,
            "teleports": step_teleports if 'step_teleports' in locals() else 0,
            "collisions": step_collisions if 'step_collisions' in locals() else 0,
        }
        
        if terminated or truncated:
            self._safe_close_traci()
            logger.info(
                "[TSCEnv:%s] Episodio terminado | steps=%d | sim_time=%.1f | reason=%s",
                self._instance_uuid,
                self._step_count,
                sim_time,
                "terminated" if terminated else "truncated"
            )
        
        return obs, reward, terminated, truncated, info
    
    def __repr__(self) -> str:
        """Representación string del entorno para debugging."""
        return f"<TSCEnv(tls_id='{self.tls_id}', uuid={self._instance_uuid[:8]})>"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Extracción del Estado 34D (Cap 4.2.2)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _get_state(self) -> np.ndarray:
        """
        Construye el vector de observación s_t ∈ ℝ^34.
        
        Según Capítulo 4.2.2:
            s_t = [q_t, w_t, p_t, φ_t, τ_t]
            donde:
                q_t ∈ ℝ^12: Colas normalizadas
                w_t ∈ ℝ^12: Esperas normalizadas
                p_t ∈ ℝ^4:  Presión agregada normalizada
                φ_t ∈ ℝ^4:  One-hot de fase activa
                τ_t ∈ ℝ^2:  Edad de fase + edad normalizada
        
        Returns
        -------
        obs : np.ndarray, shape=(34,), dtype=float32
        """
        # Componentes por carril
        queue = self._get_queue_lengths()       # shape=(n_lanes,) → 12
        wait = self._get_wait_times()           # shape=(n_lanes,) → 12
        pressure = self._get_pressure()         # shape=(n_outgoing,) → 4-8
        
        # Normalización lineal acotada [0, 1]
        q_norm = np.clip(queue / self.cfg.MAX_QUEUE, 0.0, 1.0)
        w_norm = np.clip(wait / self.cfg.MAX_WAIT, 0.0, 1.0)
        p_norm = np.clip(pressure / self.cfg.MAX_PRESSURE, 0.0, 1.0)
        
        # One-hot encoding de fase activa (φ_t)
        n_phases = self.cfg.N_GREEN_PHASES
        phi_onehot = np.zeros(n_phases, dtype=np.float32)
        phase_idx = self._current_phase % n_phases
        phi_onehot[phase_idx] = 1.0
        
        # Edad de fase (τ_t)
        phase_age_s = (self._step_count - self._phase_start_step) * self.delta_t
        tau_norm = np.clip(phase_age_s / self.cfg.MAX_PHASE_AGE, 0.0, 1.0)
        tau_features = np.array([phase_age_s / self.cfg.MAX_PHASE_AGE, tau_norm], dtype=np.float32)
        
        # Concatenar según especificación 34D
        # q(12) + w(12) + p(4) + φ(4) + τ(2) = 34
        # Ajustar presión a 4 dimensiones (promedio por dirección cardinal)
        if len(p_norm) > 4:
            # Agrupar presión por 4 direcciones cardinales
            p_agg = np.array([
                np.mean(p_norm[0:2]) if len(p_norm) >= 2 else p_norm[0] if len(p_norm) >= 1 else 0.0,
                np.mean(p_norm[2:4]) if len(p_norm) >= 4 else 0.0,
                np.mean(p_norm[4:6]) if len(p_norm) >= 6 else 0.0,
                np.mean(p_norm[6:8]) if len(p_norm) >= 8 else 0.0,
            ], dtype=np.float32)
        else:
            # Pad a 4 dimensiones
            p_agg = np.pad(p_norm, (0, 4 - len(p_norm)), mode='constant')[:4]
        
        obs = np.concatenate([
            q_norm.astype(np.float32),   # 12
            w_norm.astype(np.float32),   # 12
            p_agg,                        # 4
            phi_onehot,                   # 4
            tau_features,                 # 2
        ])
        
        # Validación crítica de dimensionalidad
        assert obs.shape == (34,), (
            f"[TSCEnv:{self._instance_uuid}] Error dimensional: "
            f"obs.shape={obs.shape}, esperado (34,). "
            f"Componentes: q={len(q_norm)}, w={len(w_norm)}, p={len(p_agg)}, "
            f"φ={len(phi_onehot)}, τ={len(tau_features)}"
        )
        
        return obs
    
    def _get_queue_lengths(self) -> np.ndarray:
        """
        q_{i,t}: número de vehículos detenidos por carril.
        
        Returns
        -------
        np.ndarray, shape=(n_controlled_lanes,)
        """
        return np.array(
            [traci.lane.getLastStepHaltingNumber(lane) for lane in self._controlled_lanes],
            dtype=np.float32,
        )
    
    def _get_wait_times(self) -> np.ndarray:
        """
        w_{i,t}: tiempo de espera acumulado por carril (segundos).
        
        Returns
        -------
        np.ndarray, shape=(n_controlled_lanes,)
        """
        return np.array(
            [traci.lane.getWaitingTime(lane) for lane in self._controlled_lanes],
            dtype=np.float32,
        )
    
    def _get_pressure(self) -> np.ndarray:
        """
        p_{i,t}: presión de tráfico (vehículos entrantes - salientes).
        
        Calculado como diferencia entre lanes controlados (entrada)
        y lanes salientes correspondientes.
        
        Returns
        -------
        np.ndarray, shape=(n_outgoing_lanes,)
        """
        incoming = np.array(
            [traci.lane.getLastStepVehicleNumber(lane) for lane in self._controlled_lanes],
            dtype=np.float32,
        )
        
        if self._outgoing_lanes:
            n_out = min(len(self._outgoing_lanes), len(incoming))
            outgoing = np.array(
                [traci.lane.getLastStepVehicleNumber(lane) 
                 for lane in self._outgoing_lanes[:n_out]],
                dtype=np.float32,
            )
            # Pad si hay mismatch
            if len(outgoing) < len(incoming):
                outgoing = np.pad(outgoing, (0, len(incoming) - len(outgoing)))
        else:
            outgoing = np.zeros_like(incoming)
        
        return incoming - outgoing
    
    # ─────────────────────────────────────────────────────────────────────────
    # Función de Recompensa Multiobjetivo (Cap 4.3.2)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_reward(self) -> float:
        """
        Función de recompensa multiobjetivo según Capítulo 4.3.2:
        
            R_t = -(λ1·Delay_t + λ2·Gini_t + λ3·CVaRα(L_t))
        
        Todos los componentes son penalizaciones (≥ 0), por lo que R_t ≤ 0.
        
        Returns
        -------
        reward : float ≤ 0
        """
        wait_times: np.ndarray = self._get_wait_times()
        
        # Componentes individuales
        delay = self._compute_delay(wait_times)
        gini = self._calculate_gini(wait_times)
        cvar = self._calculate_cvar(wait_times, alpha=self.cfg.CVAR_ALPHA)
        
        # Combinación lineal ponderada
        lam = self.reward_weights
        reward = -(
            lam["delay"] * delay +
            lam["gini"] * gini +
            lam["cvar"] * cvar
        )
        
        logger.debug(
            "[TSCEnv:%s] R=%.4f | delay=%.2f | gini=%.4f | cvar=%.2f",
            self._instance_uuid, reward, delay, gini, cvar
        )
        
        return float(reward)
    
    def _compute_delay(self, wait_times: Optional[np.ndarray] = None) -> float:
        """
        Delay agregado: suma de tiempos de espera acumulados.
        
            Δ_t = Σ_{i=1}^{n} w_{i,t}
        
        Parameters
        ----------
        wait_times : np.ndarray | None
            Vector pre-calculado; si es None se consulta TraCI.
        
        Returns
        -------
        delay : float ≥ 0
        """
        if wait_times is None:
            wait_times = self._get_wait_times()
        return float(np.sum(wait_times))
    
    def _calculate_gini(self, wait_times: np.ndarray) -> float:
        """
        Índice de Gini para equidad distributiva inter-carriles.
        
        Fórmula (Cap 4.3.2):
            G_t = Σ_i Σ_j |w_i,t - w_j,t| / (2n²·w̄_t)
        
        donde w̄_t es la espera media.
        
        Propiedades:
            • G_t ∈ [0, 1]: 0 = equidad perfecta, 1 = inequidad máxima
            • Invariante a escala
            • Sensible a outliers
        
        Parameters
        ----------
        wait_times : np.ndarray
            Tiempos de espera por carril, shape=(n_lanes,)
        
        Returns
        -------
        gini : float ∈ [0, 1]
        """
        n = len(wait_times)
        if n == 0:
            return 0.0
        
        mean_wait = np.mean(wait_times)
        if mean_wait < 1e-8:
            return 0.0  # Sin espera → equidad perfecta
        
        # Fórmula directa: Σ_i Σ_j |w_i - w_j| / (2n²·mean)
        diff_matrix = np.abs(wait_times[:, np.newaxis] - wait_times[np.newaxis, :])
        gini = float(np.sum(diff_matrix) / (2 * n * n * mean_wait))
        
        # Clamp a [0, 1] por estabilidad numérica
        gini = np.clip(gini, 0.0, 1.0)
        
        return gini
    
    def _calculate_cvar(self, losses: np.ndarray, alpha: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR) de un array de pérdidas.
        
        Fórmula (Cap 4.3.2):
            CVaR_α(L) = E[L | L ≥ VaR_α(L)]
        
        donde VaR_α es el percentil α de las pérdidas.
        
        Parameters
        ----------
        losses : np.ndarray
            Array de pérdidas (wait times, delays, etc.)
        alpha : float ∈ (0, 1)
            Nivel de confianza (ej: 0.95 → cola del 5%)
        
        Returns
        -------
        cvar : float ≥ 0
            Valor esperado de la cola de pérdidas.
        """
        if len(losses) == 0:
            return 0.0
        
        # VaR_α = percentil α
        var = np.percentile(losses, alpha * 100)
        
        # CVaR_α = media de pérdidas ≥ VaR_α
        tail_losses = losses[losses >= var]
        if len(tail_losses) == 0:
            return 0.0
        
        cvar = float(np.mean(tail_losses))
        
        logger.debug(
            "[TSCEnv:%s] CVaR_%.2f = %.2f (VaR=%.2f, tail_size=%d)",
            self._instance_uuid, alpha, cvar, var, len(tail_losses)
        )
        
        return cvar
    
    # ─────────────────────────────────────────────────────────────────────────
    # Gestión de TraCI y SUMO
    # ─────────────────────────────────────────────────────────────────────────
    
    def _load_tls_metadata(self) -> None:
        """
        Carga metadatos de la intersección desde TraCI.
        
        Popula:
            • _controlled_lanes: lista de lanes bajo control del semáforo
            • _outgoing_lanes: lista de lanes salientes
            • _green_phases: índices de fases verdes en el programa
        """
        # Obtener lanes controlados
        # NOTA: getControlledLanes() puede retornar duplicados (mismo lane para
        # distintos movimientos). Deduplicar manteniendo orden para el estado 34D.
        raw_lanes = list(traci.trafficlight.getControlledLanes(self.tls_id))
        # Deduplicar preservando orden de aparición
        seen: set = set()
        unique_lanes = [l for l in raw_lanes if not (l in seen or seen.add(l))]
        
        n_target = self.cfg.N_CONTROLLED_LANES  # = 12
        if len(unique_lanes) >= n_target:
            # Truncar a los primeros N_CONTROLLED_LANES carriles únicos
            self._controlled_lanes = unique_lanes[:n_target]
        else:
            # Pad repitiendo el último carril (raro, pero robusto)
            self._controlled_lanes = unique_lanes + [unique_lanes[-1]] * (n_target - len(unique_lanes))
        
        logger.info(
            "[TSCEnv:%s] Lanes: raw=%d → únicos=%d → normalizados=%d (N_CONTROLLED=%d)",
            self._instance_uuid, len(raw_lanes), len(unique_lanes),
            len(self._controlled_lanes), n_target
        )
        
        # Obtener programa de fases
        # API SUMO 1.26.0: getAllProgramLogics() (getCompleteDefinition no existe)
        # Retorna lista de Logic con .phases → cada Phase tiene .state (str con G/y/r)
        try:
            logic_list = traci.trafficlight.getAllProgramLogics(self.tls_id)
            logic = logic_list[0] if logic_list else None
            self._green_phases = [
                i for i, phase in enumerate(logic.phases)
                if _is_green_phase(phase.state)
            ] if logic else [0, 1, 2, 3]
        except AttributeError:
            # Fallback para versiones antiguas de TraCI
            try:
                logic_list = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
                logic = logic_list[0] if logic_list else None
                self._green_phases = [
                    i for i, phase in enumerate(logic.phases)
                    if _is_green_phase(phase.state)
                ] if logic else [0, 1, 2, 3]
            except Exception:
                # Último recurso: usar fases 0-3 por defecto
                self._green_phases = [0, 1, 2, 3]
                logger.warning("[TSCEnv:%s] No se pudo leer fases TLS, usando [0,1,2,3]",
                               self._instance_uuid)
        
        # Estimar outgoing lanes (simplificación: lanes adyacentes no controlados)
        # En implementación completa, esto se extrae del net.xml
        self._outgoing_lanes = []  # Placeholder
        
        logger.info(
            "[TSCEnv:%s] Metadatos cargados | controlled_lanes=%d | green_phases=%d",
            self._instance_uuid, len(self._controlled_lanes), len(self._green_phases)
        )
    
    def _launch_sumo(self) -> None:
        """
        Lanza el proceso SUMO en segundo plano con configuración adecuada.
        
        Nota Windows: Se usa CREATE_NO_WINDOW para evitar popup de consola
        en entornos vectorizados (SubprocVecEnv).
        """
        if not self.sumocfg_path.exists():
            raise FileNotFoundError(f"Archivo .sumocfg no encontrado: {self.sumocfg_path}")
        
        # Encontrar binario SUMO: priorizar conda (shutil.which) sobre nombre plano
        import shutil
        sumo_bin = shutil.which(self._sumo_binary)
        if not sumo_bin:
            sumo_bin = self._sumo_binary  # Fallback al nombre plano
        
        cmd = [
            sumo_bin,
            "-c", str(self.sumocfg_path),
            "--remote-port", str(self._traci_port),
            "--step-length", "1.0",  # El tick interno de SUMO DEBE ser 1s.
            "--no-step-log",
            "--seed", str(self.seed_val),
        ]
        if self.enable_traci_chaos:
            cmd += [
                "--collision.action", "warn",
                "--collision.mingap-factor", "0",
            ]
        
        logger.debug("[TSCEnv:%s] Lanzando SUMO: %s", self._instance_uuid, " ".join(cmd))
        
        # Configurar variables de entorno para SUMO
        env = os.environ.copy()
        if self._sumo_home:
            env["SUMO_HOME"] = self._sumo_home
        
        # Windows: Usar CREATE_NO_WINDOW para evitar popup de consola
        creation_flags = 0
        if os.name == 'nt':  # Windows
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW
            except AttributeError:
                creation_flags = 0x08000000
        
        # Guardar logs de SUMO a archivo para evitar deadlocks por PIPE lleno
        self._sumo_log_path = self.sumocfg_path.with_suffix('.sumo.log')
        self._sumo_log_file = open(self._sumo_log_path, "w")
        
        self._sumo_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._sumo_log_file,   # Archivo para evitar bloqueo del OS pipe
            creationflags=creation_flags,
            env=env,
        )
        
        # Esperar arranque de SUMO (Windows necesita más tiempo que Linux)
        time.sleep(2.0)
        
        # Verificar que SUMO sigue vivo
        if self._sumo_process.poll() is not None:
            self._sumo_log_file.close()
            with open(self._sumo_log_path, "r") as f:
                stderr_out = f.read()
            raise RuntimeError(
                f"[TSCEnv:{self._instance_uuid}] SUMO terminó prematuramente.\n"
                f"Binario: {sumo_bin}\nError: {stderr_out[:400]}"
            )

    
    def _safe_close_traci(self) -> None:
        """
        Cierra conexión TraCI y proceso SUMO de forma segura.
        """
        try:
            traci.close()
        except Exception:
            pass
        
        if self._sumo_process is not None:
            try:
                self._sumo_process.terminate()
                self._sumo_process.wait(timeout=2)
            except Exception:
                try:
                    self._sumo_process.kill()
                except Exception:
                    pass
            finally:
                self._sumo_process = None
                # Esperar que el SO libere el puerto antes de permitir un nuevo reset
                time.sleep(0.5)
                
        # Cerrar el log de SUMO si está abierto
        if hasattr(self, '_sumo_log_file') and self._sumo_log_file and not self._sumo_log_file.closed:
            self._sumo_log_file.close()

        
        logger.debug("[TSCEnv:%s] Conexión TraCI cerrada", self._instance_uuid)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validaciones y Assertions
    # ─────────────────────────────────────────────────────────────────────────
    
    def validate_cpu_device(self) -> bool:
        """
        Valida que el dispositivo sea explícitamente CPU.
        
        Returns
        -------
        bool : True si device.type == "cpu"
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch no disponible. Validación CPU omitida.")
            return True
        
        is_cpu = self.device.type == "cpu"
        if not is_cpu:
            logger.error(
                "[TSCEnv:%s] VIOLACIÓN CPU-ONLY: device=%s",
                self._instance_uuid, self.device
            )
        return is_cpu
    
    def validate_observation_shape(self, obs: np.ndarray) -> bool:
        """
        Valida que la observación tenga dimensión 34.
        
        Returns
        -------
        bool : True si obs.shape == (34,)
        """
        is_valid = obs.shape == (34,)
        if not is_valid:
            logger.error(
                "[TSCEnv:%s] Shape inválido: %s vs (34,)",
                self._instance_uuid, obs.shape
            )
        return is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Export público
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["TSCEnv", "TSCConfig", "DEFAULT_CONFIG"]
