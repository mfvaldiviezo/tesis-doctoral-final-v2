"""
sumo_env.py — Entorno Gymnasium personalizado para Control Semafórico
=====================================================================
Integra Eclipse SUMO vía TraCI para ofrecer una interfaz RL estándar.

Espacio de estados (por carril controlado):
    q_i   — Longitud de cola (vehículos detenidos)
    w_i   — Tiempo de espera acumulado (segundos)
    p_i   — Presión de tráfico (entrantes − salientes)
    φ     — Índice de fase activa (normalizado)
    τ     — Edad de la fase activa (segundos)

Espacio de acciones:
    Discrete(n_green_phases) — Selección de la siguiente fase verde

Función de recompensa multiobjetivo:
    R_t = −λ1·Delay − λ2·Pressure − λ3·Gini − λ4·CVaR_α(L_t)
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# TraCI se importa de forma lazy para facilitar tests sin SUMO instalado
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _find_free_port(start: int = 8813, max_tries: int = 100) -> int:
    """Encuentra un puerto TCP libre a partir de `start`."""
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No se encontró puerto libre en el rango [{start}, {start + max_tries})")


def _is_green_phase(phase_state: str) -> bool:
    """Retorna True si la fase contiene al menos un movimiento verde (G/g)."""
    return any(c in ("G", "g") for c in phase_state)


# ─────────────────────────────────────────────────────────────────────────────
# Entorno principal
# ─────────────────────────────────────────────────────────────────────────────

class SumoRLEnv(gym.Env):
    """
    Entorno Gymnasium para control semafórico de una intersección con SUMO+TraCI.

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
        Pesos λ1..λ4 de la función multiobjetivo.
    sumo_home : str | None
        Directorio de instalación de SUMO. Si es None, usa $SUMO_HOME.
    seed : int
        Semilla para reproducibilidad.
    """

    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    # Bounds para normalización del espacio de observación
    MAX_QUEUE: float = 50.0       # vehículos máx por carril
    MAX_WAIT: float = 300.0       # segundos máx de espera
    MAX_PRESSURE: float = 50.0    # diferencia máx de vehículos
    MAX_PHASE_AGE: float = 120.0  # segundos máx en una fase

    def __init__(
        self,
        sumocfg_path: str | Path,
        tls_id: str,
        use_gui: bool = False,
        delta_t: int = 5,
        max_steps: int = 3600,
        reward_weights: Optional[Dict[str, float]] = None,
        sumo_home: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if not TRACI_AVAILABLE:
            raise ImportError(
                "TraCI no está disponible. Instala SUMO: pip install eclipse-sumo traci sumolib"
            )

        # ── Configuración ────────────────────────────────────────────────────
        self.sumocfg_path = Path(sumocfg_path)
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.delta_t = delta_t
        self.max_steps = max_steps
        self.seed_val = seed

        self._rng = np.random.default_rng(seed)

        # Pesos de la recompensa multiobjetivo
        _default_weights = {"delay": 0.4, "pressure": 0.3, "gini": 0.15, "cvar": 0.15}
        self.reward_weights: Dict[str, float] = {**_default_weights, **(reward_weights or {})}

        # SUMO_HOME
        if sumo_home:
            os.environ["SUMO_HOME"] = sumo_home
        self._sumo_home = os.environ.get("SUMO_HOME", "")
        self._sumo_binary = "sumo-gui" if use_gui else "sumo"

        # ── Estado interno (se rellena en reset) ────────────────────────────
        self._traci_port: int = -1
        self._sumo_process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._step_count: int = 0
        self._phase_start_step: int = 0
        self._current_phase: int = 0

        # Metadatos de la intersección (cargados al conectar TraCI)
        self._controlled_lanes: List[str] = []
        self._outgoing_lanes: List[str] = []
        self._green_phases: List[int] = []   # índices de fases verdes
        self._n_lanes: int = 0

        # Historial de longitudes de cola (para CVaR)
        self._queue_history: List[np.ndarray] = []

        # ── Espacios (se definen definitivamente en _init_spaces) ───────────
        # Dimensión provisional; se recalculará en reset() al conocer n_lanes
        self._obs_dim: int = 1  # placeholder
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1)  # placeholder

        logger.info(
            "SumoRLEnv inicializado | cfg=%s | tls=%s | δt=%ds",
            self.sumocfg_path.name,
            self.tls_id,
            self.delta_t,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de vida
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia la simulación y retorna (observation, info).

        Cierra cualquier sesión TraCI previa, lanza un nuevo proceso SUMO
        en un puerto libre y obtiene los metadatos de la intersección.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Cerrar episodio anterior si existe
        self._safe_close_traci()

        # Seleccionar puerto libre y lanzar SUMO
        self._traci_port = _find_free_port()
        self._launch_sumo()

        # Conectar TraCI
        traci.init(port=self._traci_port, numRetries=10)
        logger.debug("TraCI conectado en puerto %d", self._traci_port)

        # Obtener metadatos de la intersección
        self._load_tls_metadata()

        # Definir espacios con dimensiones reales
        self._init_spaces()

        # Reiniciar contadores
        self._step_count = 0
        self._phase_start_step = 0
        self._current_phase = traci.trafficlight.getPhase(self.tls_id)
        self._queue_history = []

        obs = self._get_state()
        info: Dict[str, Any] = {
            "tls_id": self.tls_id,
            "n_lanes": self._n_lanes,
            "n_green_phases": len(self._green_phases),
            "port": self._traci_port,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Aplica la acción (cambio de fase), avanza la simulación δt segundos
        y retorna (obs, reward, terminated, truncated, info).

        Parameters
        ----------
        action : int
            Índice dentro de self._green_phases (fase verde a activar).
        """
        assert self.action_space.contains(action), f"Acción inválida: {action}"

        # Convertir acción discreta → índice de fase en SUMO
        target_phase: int = self._green_phases[int(action)]

        # Cambiar fase si es distinta a la actual
        if target_phase != self._current_phase:
            # En SUMO se recomienda insertar una fase amarilla antes,
            # pero para simplificar el andamiaje cambiamos directamente.
            traci.trafficlight.setPhase(self.tls_id, target_phase)
            self._phase_start_step = self._step_count
            self._current_phase = target_phase

        # Avanzar simulación delta_t pasos
        for _ in range(self.delta_t):
            traci.simulationStep()

        self._step_count += 1

        # Recolectar nuevo estado
        obs = self._get_state()

        # Guardar colas para CVaR
        queue_vec = self._get_queue_lengths()
        self._queue_history.append(queue_vec)

        # Calcular recompensa
        reward = self._calculate_reward()

        # Condiciones de terminación
        sim_time = traci.simulation.getTime()
        sim_end  = traci.simulation.getEndTime()
        terminated: bool = sim_time >= sim_end
        truncated: bool  = self._step_count >= self.max_steps

        info: Dict[str, Any] = {
            "step": self._step_count,
            "sim_time": sim_time,
            "phase": self._current_phase,
            "phase_age": (self._step_count - self._phase_start_step) * self.delta_t,
            "total_queue": float(queue_vec.sum()),
        }

        if terminated or truncated:
            self._safe_close_traci()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Cierra la conexión TraCI y el proceso SUMO de forma segura."""
        self._safe_close_traci()
        logger.info("SumoRLEnv cerrado correctamente.")

    # ─────────────────────────────────────────────────────────────────────────
    # Espacio de observación y acción
    # ─────────────────────────────────────────────────────────────────────────

    def _init_spaces(self) -> None:
        """
        Define observation_space y action_space con las dimensiones reales
        obtenidas de la intersección en SUMO.

        Observación (por carril): [q_i, w_i, p_i]  → n_lanes * 3 features
        Observación global:       [φ, τ]            → 2 features
        Total:                    n_lanes * 3 + 2
        """
        self._obs_dim = self._n_lanes * 3 + 2

        self.observation_space = spaces.Box(
            low=np.zeros(self._obs_dim, dtype=np.float32),
            high=np.ones(self._obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # Una acción por cada fase verde disponible
        n_green = max(len(self._green_phases), 1)
        self.action_space = spaces.Discrete(n_green)

        logger.debug(
            "Espacios inicializados | obs_dim=%d | n_green_phases=%d",
            self._obs_dim,
            n_green,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Extracción del estado  _get_state()
    # ─────────────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Construye el vector de observación normalizado al rango [0, 1].

        Returns
        -------
        np.ndarray, shape=(obs_dim,), dtype=float32
        """
        queue    = self._get_queue_lengths()          # (n_lanes,)
        wait     = self._get_wait_times()             # (n_lanes,)
        pressure = self._get_pressure()               # (n_lanes,)

        # Normalizar por valores máximos esperados
        q_norm = np.clip(queue    / self.MAX_QUEUE,    0.0, 1.0)
        w_norm = np.clip(wait     / self.MAX_WAIT,     0.0, 1.0)
        p_norm = np.clip(pressure / self.MAX_PRESSURE, 0.0, 1.0)

        # Variables globales de fase
        n_phases = max(len(self._green_phases), 1)
        phi_norm = self._current_phase / n_phases                    # φ ∈ [0,1]

        phase_age_s = (self._step_count - self._phase_start_step) * self.delta_t
        tau_norm = np.clip(phase_age_s / self.MAX_PHASE_AGE, 0.0, 1.0)  # τ ∈ [0,1]

        # Concatenar: [q_0,w_0,p_0, q_1,w_1,p_1, ..., φ, τ]
        lane_features = np.stack([q_norm, w_norm, p_norm], axis=1).flatten()
        obs = np.concatenate([lane_features, [phi_norm, tau_norm]]).astype(np.float32)

        assert obs.shape == (self._obs_dim,), (
            f"Shape inesperado: {obs.shape} vs esperado ({self._obs_dim},)"
        )
        return obs

    def _get_queue_lengths(self) -> np.ndarray:
        """q_{i,t}: número de vehículos detenidos por carril."""
        return np.array(
            [traci.lane.getLastStepHaltingNumber(lane) for lane in self._controlled_lanes],
            dtype=np.float32,
        )

    def _get_wait_times(self) -> np.ndarray:
        """w_{i,t}: tiempo de espera acumulado por carril (segundos)."""
        return np.array(
            [traci.lane.getWaitingTime(lane) for lane in self._controlled_lanes],
            dtype=np.float32,
        )

    def _get_pressure(self) -> np.ndarray:
        """
        p_{i,t} = vehículos entrantes − vehículos salientes por carril.

        Aproximación: vehículos en carril entrante menos vehículos
        en el carril de salida correspondiente (por posición en lista).
        """
        incoming = np.array(
            [traci.lane.getLastStepVehicleNumber(lane) for lane in self._controlled_lanes],
            dtype=np.float32,
        )
        if self._outgoing_lanes:
            n_out = min(len(self._outgoing_lanes), len(self._controlled_lanes))
            outgoing = np.array(
                [traci.lane.getLastStepVehicleNumber(lane) for lane in self._outgoing_lanes[:n_out]],
                dtype=np.float32,
            )
            # Pad si hay más carriles entrantes que salientes
            if len(outgoing) < len(incoming):
                outgoing = np.pad(outgoing, (0, len(incoming) - len(outgoing)))
        else:
            outgoing = np.zeros_like(incoming)

        return incoming - outgoing

    # ─────────────────────────────────────────────────────────────────────────
    # Función de recompensa  _calculate_reward()
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        """
        Función de recompensa multiobjetivo:

            R_t = −(λ1·Delay + λ2·Pressure + λ3·Gini + λ4·CVaR_α(L_t))

        Todos los componentes son penalizaciones (≥ 0), por lo que R_t ≤ 0.
        Los pesos λ_k se leen de ``self.reward_weights`` y deben sumar 1.

        Returns
        -------
        float
            Recompensa escalar (siempre ≤ 0).
        """
        wait_times: np.ndarray = self._get_wait_times()   # w_{i,t} shape=(n_lanes,)

        delay    = self._compute_delay(wait_times)
        pressure = self._compute_pressure_scalar()
        gini     = self._calculate_gini(wait_times)
        cvar     = self._calculate_cvar(wait_times, alpha=0.90)

        lam = self.reward_weights
        reward = -(
            lam["delay"]    * delay    +
            lam["pressure"] * pressure +
            lam["gini"]     * gini     +
            lam["cvar"]     * cvar
        )

        logger.debug(
            "R=%.3f | delay=%.2f | pressure=%.2f | gini=%.4f | cvar=%.2f",
            reward, delay, pressure, gini, cvar,
        )
        return float(reward)

    def _compute_delay(self, wait_times: Optional[np.ndarray] = None) -> float:
        """
        Delay agregado: suma de tiempos de espera acumulados por carril.

            Δ_t = Σ_{i=1}^{n}  w_{i,t}      [segundos]

        Parameters
        ----------
        wait_times : np.ndarray | None
            Vector pre-calculado; si es None se consulta TraCI de nuevo.
        """
        if wait_times is None:
            wait_times = self._get_wait_times()
        return float(wait_times.sum())

    def _compute_pressure_scalar(self) -> float:
        """
        Presión total: suma del valor absoluto de la presión por carril.

            P_t = Σ_{i=1}^{n}  |entrantes_i − salientes_i|
        """
        return float(np.abs(self._get_pressure()).sum())

    # ── Métricas de riesgo y equidad ─────────────────────────────────────────

    @staticmethod
    def _calculate_gini(wait_times: np.ndarray) -> float:
        """
        Índice de Gini espacial sobre los tiempos de espera por carril.

        Cuantifica la inequidad distributiva entre carriles: G = 0 indica
        distribución perfectamente equitativa; G → 1 indica concentración
        extrema del tiempo de espera en un único carril.

        Formulación vectorizada (O(n log n) por el sort):

            G = (2 · Σ_{i=1}^{n}  i · w_{(i)}) / (n · Σ_{i=1}^{n} w_i)  −  (n+1)/n

        Donde w_{(i)} son los tiempos ordenados de menor (i=1) a mayor (i=n).

        Parameters
        ----------
        wait_times : np.ndarray, shape=(n,)
            Tiempos de espera acumulados por carril w_{i,t} ≥ 0.

        Returns
        -------
        float
            Índice de Gini G ∈ [0, 1].  Retorna 0.0 si la suma total es 0
            (todos los carriles están libres — equidad perfecta).
        """
        n: int = len(wait_times)
        if n == 0:
            return 0.0

        total: float = float(wait_times.sum())
        if total == 0.0:
            # Todos los carriles con w=0: equidad máxima → G = 0
            return 0.0

        sorted_w: np.ndarray = np.sort(wait_times)          # w_{(1)} ≤ ... ≤ w_{(n)}
        ranks: np.ndarray    = np.arange(1, n + 1, dtype=np.float64)  # i = 1..n

        # G = 2·Σ(i·w_(i)) / (n·Σw_i)  −  (n+1)/n
        gini: float = (
            2.0 * float(np.dot(ranks, sorted_w)) / (n * total)
        ) - (n + 1.0) / n

        # Clamp numérico: G ∈ [0, 1] por construcción matemática,
        # pero pequeños errores de punto flotante pueden sacarlo levemente.
        return float(np.clip(gini, 0.0, 1.0))

    @staticmethod
    def _calculate_cvar(losses: np.ndarray, alpha: float = 0.90) -> float:
        """
        CVaR_α (Conditional Value at Risk) sobre la distribución de pérdidas.

        El CVaR cuantifica el riesgo de cola: el valor esperado de las pérdidas
        que superan el cuantil α (los peores (1−α)·100 % de casos).

            VaR_α   = percentil α de {L_i}     → np.percentile(losses, α·100)
            CVaR_α  = E[L_i | L_i ≥ VaR_α]    → media del conjunto de cola

        En este entorno, ``losses`` es el vector de tiempos de espera por
        carril w_{i,t}: un carril con espera muy alta representa una pérdida
        grave de servicio (riesgo de cola).

        Parameters
        ----------
        losses : np.ndarray, shape=(n,)
            Vector de pérdidas L_{i,t} ≥ 0 (típicamente wait_times).
        alpha : float
            Nivel de confianza α ∈ (0, 1).  Default = 0.90 (CVaR al 90 %).

        Returns
        -------
        float
            CVaR_α ≥ 0.  Retorna 0.0 si el vector está vacío o si ningún
            elemento supera el VaR (cola vacía).
        """
        if len(losses) == 0:
            return 0.0

        var_alpha: float = float(np.percentile(losses, alpha * 100.0))

        # Seleccionar pérdidas en la cola superior (≥ VaR_α)
        tail: np.ndarray = losses[losses >= var_alpha]

        if len(tail) == 0:
            return 0.0

        return float(np.mean(tail))

    # ─────────────────────────────────────────────────────────────────────────
    # Métodos de soporte interno
    # ─────────────────────────────────────────────────────────────────────────

    def _launch_sumo(self) -> None:
        """Lanza el proceso SUMO con TraCI habilitado en el puerto seleccionado."""
        if not self.sumocfg_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo SUMO: {self.sumocfg_path}")

        sumo_cmd: List[str] = [
            self._sumo_binary,
            "-c", str(self.sumocfg_path),
            "--remote-port", str(self._traci_port),
            "--seed", str(self.seed_val),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",    # desactiva teleport de vehículos
        ]

        logger.debug("Lanzando SUMO: %s", " ".join(sumo_cmd))
        try:
            self._sumo_process = subprocess.Popen(
                sumo_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.5)  # Dar tiempo a SUMO para inicializar el socket
        except FileNotFoundError as exc:
            raise EnvironmentError(
                f"Binario SUMO '{self._sumo_binary}' no encontrado. "
                f"Verifica $SUMO_HOME o que SUMO esté en el PATH. ({exc})"
            ) from exc

    def _load_tls_metadata(self) -> None:
        """
        Carga los metadatos del semáforo (carriles controlados, fases verdes)
        desde TraCI tras establecer la conexión.
        """
        # Carriles entrantes controlados por el semáforo
        self._controlled_lanes = list(
            dict.fromkeys(traci.trafficlight.getControlledLanes(self.tls_id))
        )  # dict.fromkeys preserva orden y elimina duplicados

        # Carriles salientes (heurística: salidas del nodo de la intersección)
        try:
            links = traci.trafficlight.getControlledLinks(self.tls_id)
            outgoing: List[str] = []
            for link_group in links:
                for link in link_group:
                    # link = (from_lane, to_lane, via_lane)
                    if len(link) >= 2 and link[1] and link[1] not in outgoing:
                        outgoing.append(link[1])
            self._outgoing_lanes = outgoing
        except traci.TraCIException:
            self._outgoing_lanes = []

        self._n_lanes = len(self._controlled_lanes)
        if self._n_lanes == 0:
            raise ValueError(
                f"El semáforo '{self.tls_id}' no controla ningún carril. "
                "Verifica el tls_id en tu archivo .sumocfg."
            )

        # Identificar fases verdes en el programa del semáforo
        program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self._green_phases = [
            idx for idx, phase in enumerate(program.phases)
            if _is_green_phase(phase.state)
        ]
        if not self._green_phases:
            raise ValueError(
                f"No se encontraron fases verdes para el semáforo '{self.tls_id}'."
            )

        logger.info(
            "TLS '%s' | carriles_ctrl=%d | salientes=%d | fases_verdes=%s",
            self.tls_id,
            self._n_lanes,
            len(self._outgoing_lanes),
            self._green_phases,
        )

    def _safe_close_traci(self) -> None:
        """Cierra la conexión TraCI y el proceso SUMO sin lanzar excepciones."""
        try:
            if traci.isLoaded():
                traci.close()
                logger.debug("Conexión TraCI cerrada.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error al cerrar TraCI: %s", exc)

        if self._sumo_process is not None:
            try:
                self._sumo_process.terminate()
                self._sumo_process.wait(timeout=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error al terminar proceso SUMO: %s", exc)
            finally:
                self._sumo_process = None

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades de inspección
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def n_lanes(self) -> int:
        """Número de carriles controlados por el semáforo."""
        return self._n_lanes

    @property
    def n_green_phases(self) -> int:
        """Número de fases verdes disponibles (= tamaño del espacio de acciones)."""
        return len(self._green_phases)

    @property
    def controlled_lanes(self) -> List[str]:
        """IDs de los carriles controlados (copia defensiva)."""
        return list(self._controlled_lanes)

    def __repr__(self) -> str:
        return (
            f"SumoRLEnv(cfg={self.sumocfg_path.name!r}, "
            f"tls={self.tls_id!r}, "
            f"δt={self.delta_t}s, "
            f"obs_dim={self._obs_dim}, "
            f"n_actions={self.action_space.n})"
        )
