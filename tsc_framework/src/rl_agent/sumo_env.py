"""
sumo_env.py — TrafficEnv
========================
Entorno Gymnasium para el Módulo 3 del framework de tesis:
Control semafórico inteligente sobre la red real de Hangzhou 4×4,
usando escenarios de estrés probabilísticos generados por Vine Copula.

Diferenciador respecto a src/rl_env/sumo_env.py (SumoRLEnv):
    • Selecciona aleatoriamente un .rou.xml de sumo_configs/routes/ en cada reset().
    • Gestiona el ciclo completo de vida TraCI (inicio/parada) sin necesidad
      de un .sumocfg preexistente — lo genera en tiempo de ejecución.
    • Expone un espacio de observación compacto (densidad + espera por carril)
      optimizado para la intersección central del grid 4×4.

Espacio de estados  (Box, float32):
    [ queue_lane_0, …, queue_lane_N,        # vehículos detenidos por carril
      wait_lane_0,  …, wait_lane_N,         # tiempo de espera acumulado (s)
      phase_idx_norm,                        # fase activa normalizada ∈ [0,1]
      phase_age_norm ]                       # edad de la fase (s / MAX_AGE)

Espacio de acciones (Discrete):
    k  ∈ {0, …, n_green_phases − 1}  → selección de la siguiente fase verde

Función de recompensa:
    R = −Total_Wait − ω·Gini(wait_times)
    donde ω = gini_penalty_weight (configurable).
"""

from __future__ import annotations

import logging
import os
import random
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# TraCI y sumolib: importación lazy para permitir tests sin SUMO instalado
try:
    import traci
    import traci.constants as tc
    import sumolib
    _TRACI_OK = True
except ImportError:
    _TRACI_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas por defecto del proyecto
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NET_XML_DEFAULT = (
    _PROJECT_ROOT
    / "data" / "raw" / "macro_traffic" / "Hangzhou" / "4_4"
    / "hangzhou_4x4_gudang_18041610_1h.net.xml"
)
_ROUTES_DIR_DEFAULT = _PROJECT_ROOT / "sumo_configs" / "routes"

# ---------------------------------------------------------------------------
# Constantes de simulación
# ---------------------------------------------------------------------------
_DELTA_SIM_SECONDS: int = 5          # segundos de simulación por step()
_MAX_SIM_SECONDS: int = 3_600        # duración máxima de un episodio (1 h)
_MAX_PHASE_AGE: float = 120.0        # para normalizar la edad de la fase
_MAX_WAIT: float = 300.0             # para normalizar tiempos de espera (s)
_MAX_QUEUE: float = 50.0             # para normalizar longitud de cola (veh)
_SUMO_PORT_BASE: int = 8813          # puerto base para TraCI


# ===========================================================================
# Utilidades privadas
# ===========================================================================

def _find_free_port(base: int = _SUMO_PORT_BASE, max_tries: int = 100) -> int:
    """Busca un puerto TCP libre a partir de ``base``."""
    for port in range(base, base + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No se encontró puerto libre en [{base}, {base + max_tries})")


def _is_green_phase(state: str) -> bool:
    """Devuelve True si la cadena de fase contiene al menos un verde (G/g)."""
    return any(c in ("G", "g") for c in state)


def _write_temp_sumocfg(net_xml: Path, rou_xml: Path, out_dir: Path) -> Path:
    """
    Genera un archivo .sumocfg mínimo en disco para iniciar SUMO+TraCI.

    SUMO requiere siempre un archivo de configuración; como los escenarios
    de estrés son generados dinámicamente, creamos el .sumocfg en un
    directorio temporal en cada episodio.

    Args:
        net_xml: Ruta absoluta al archivo de red .net.xml.
        rou_xml: Ruta absoluta al archivo de rutas .rou.xml del episodio.
        out_dir: Directorio donde se escribe el .sumocfg temporal.

    Returns:
        Path al .sumocfg generado.
    """
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_xml.as_posix()}"/>
        <route-files value="{rou_xml.as_posix()}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{_MAX_SIM_SECONDS}"/>
    </time>
</configuration>
"""
    cfg_path = out_dir / f"episode_{rou_xml.stem}.sumocfg"
    cfg_path.write_text(cfg_content, encoding="utf-8")
    return cfg_path


# ===========================================================================
# Entorno principal
# ===========================================================================

class TrafficEnv(gym.Env):
    """
    Entorno Gymnasium para control semafórico de la intersección central
    de la red Hangzhou 4×4 con escenarios de estrés probabilísticos.

    Parameters
    ----------
    net_xml : Path | str, optional
        Ruta al archivo de red .net.xml de Hangzhou.
        Por defecto: ``data/raw/macro_traffic/Hangzhou/4_4/*.net.xml``.
    routes_dir : Path | str, optional
        Directorio con los .rou.xml generados por route_generator.py.
        Por defecto: ``sumo_configs/routes/``.
    tls_id : str, optional
        ID del semáforo a controlar. Por defecto ``"J2"`` (intersección
        central del grid 4×4 de Hangzhou).
    delta_time : int, optional
        Segundos de simulación por cada llamada a step(). Default 5.
    gini_penalty_weight : float, optional
        Peso ω de la penalización por equidad Gini en la recompensa. Default 50.0.
    use_gui : bool, optional
        Si True arranca ``sumo-gui`` en lugar de ``sumo``. Default False.
    seed : int | None, optional
        Semilla de aleatoriedad para reproducibilidad de reset(). Default None.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        net_xml: Path | str = _NET_XML_DEFAULT,
        routes_dir: Path | str = _ROUTES_DIR_DEFAULT,
        tls_id: str = "intersection_2_2",
        delta_time: int = _DELTA_SIM_SECONDS,
        gini_penalty_weight: float = 50.0,
        use_gui: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.net_xml = Path(net_xml)
        self.routes_dir = Path(routes_dir)
        self.tls_id = tls_id
        self.delta_time = delta_time
        self.gini_penalty_weight = gini_penalty_weight
        self.use_gui = use_gui

        # Semilla interna
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Estado de TraCI — etiqueta única por instancia para aislamiento en paralelo
        self.label: str = f"env_{uuid.uuid4().hex}"
        self.traci_conn: Any = None          # conexión explícita obtenida con getConnection()
        self._traci_port: int | None = None
        self._sumo_proc: subprocess.Popen | None = None
        self._connected: bool = False

        # Metadatos de la intersección (se llenan en _init_intersection)
        self._controlled_lanes: list[str] = []
        self._green_phases: list[int] = []      # índices de fases verdes
        self._n_phases: int = 0
        self._current_phase: int = 0
        self._phase_age: float = 0.0
        self._sim_step: int = 0

        # Directorio temporal para .sumocfg generados dinámicamente
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="tsc_env_"))

        # Validaciones de archivos
        if not self.net_xml.exists():
            raise FileNotFoundError(
                f"Red SUMO no encontrada: {self.net_xml}\n"
                "Verifica que los datos macro estén descargados."
            )
        if not self.routes_dir.exists():
            raise FileNotFoundError(
                f"Directorio de rutas no encontrado: {self.routes_dir}\n"
                "Ejecuta primero: python src/sumo_configs/route_generator.py"
            )

        # Obtener lista de escenarios disponibles
        self._rou_files: list[Path] = sorted(self.routes_dir.glob("*.rou.xml"))
        if not self._rou_files:
            raise FileNotFoundError(
                f"No se encontraron archivos .rou.xml en {self.routes_dir}"
            )
        logger.info(f"TrafficEnv: {len(self._rou_files)} escenarios disponibles.")

        # ── Espacios de observación y acción (FIJOS — nunca se redimensionan) ──────
        # SB3 construye la red neuronal desde observation_space en __init__.
        # Si se redefiniera en reset()/_init_intersection(), causaría shape mismatch.
        # Fijamos 16 carriles (estimación 4x4 grid) y 4 fases verdes como máximos.
        # _get_observation() siempre rellena con ceros hasta _OBS_DIM_FIXED.
        _N_LANES_MAX = 16
        self._obs_dim_fixed: int = _N_LANES_MAX * 2 + 2  # 34
        self._n_green_phases_fixed: int = 4               # fases verdes típicas

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim_fixed,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._n_green_phases_fixed)

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reinicia el entorno seleccionando un escenario de estrés aleatoriamente.

        Flujo:
            1. Cierra la conexión TraCI anterior (si existe).
            2. Elige un .rou.xml aleatorio de sumo_configs/routes/.
            3. Genera el .sumocfg temporal correspondiente.
            4. Arranca SUMO en modo CLI (``sumo -W``) y conecta TraCI.
            5. Inicializa los metadatos de la intersección controlada.
            6. Retorna la observación inicial.

        Args:
            seed: Semilla para re-inicializar el RNG de selección de escenario.
            options: Opciones adicionales (reservado para uso futuro).

        Returns:
            (obs, info) — Observación inicial y diccionario de info.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        # Cerrar sesión TraCI anterior
        self._close_traci()

        # Selección aleatoria del escenario de estrés
        rou_xml = self._rng.choice(self._rou_files)
        logger.info(f"  [reset] Escenario seleccionado: {rou_xml.name}")

        # Generar .sumocfg temporal
        cfg_path = _write_temp_sumocfg(self.net_xml, rou_xml, self._tmp_dir)

        # Arrancar SUMO y conectar TraCI
        self._start_traci(cfg_path)

        # Inicializar metadatos de la intersección
        self._init_intersection()

        # Resetear contadores de episodio
        self._sim_step = 0
        self._phase_age = 0.0
        self._current_phase = self._green_phases[0] if self._green_phases else 0

        obs = self._get_observation()
        info: dict = {
            "scenario": rou_xml.name,
            "tls_id": self.tls_id,
            "n_controlled_lanes": len(self._controlled_lanes),
            "n_green_phases": len(self._green_phases),
        }
        return obs, info

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Ejecuta un paso de control semafórico:

            1. Mapea ``action`` → índice de fase verde real en SUMO.
            2. Aplica la fase con traci.trafficlight.setPhase().
            3. Avanza ``delta_time`` segundos de simulación.
            4. Calcula la recompensa basada en equidad y eficiencia.
            5. Retorna (obs, reward, terminated, truncated, info).

        Args:
            action: Entero en [0, n_green_phases) seleccionado por el agente.

        Returns:
            Tupla (observación, recompensa, terminado, truncado, info).
        """
        if not self._connected:
            raise RuntimeError("step() llamado sin conexión TraCI activa. Llama reset() primero.")

        # ── Acción: seleccionar fase verde correspondiente ─────────────────
        action = int(np.clip(action, 0, len(self._green_phases) - 1))
        target_phase = self._green_phases[action]

        try:
            self.traci_conn.trafficlight.setPhase(self.tls_id, target_phase)
        except traci.exceptions.TraCIException as exc:
            logger.warning(f"  setPhase falló: {exc} — manteniendo fase actual.")

        # Actualizar edad de la fase
        if target_phase == self._current_phase:
            self._phase_age += self.delta_time
        else:
            self._phase_age = 0.0
            self._current_phase = target_phase

        # ── Avanzar simulación delta_time segundos ─────────────────────────
        for _ in range(self.delta_time):
            self.traci_conn.simulationStep()
            self._sim_step += 1

        # ── Calcular recompensa ────────────────────────────────────────────
        wait_times = self._get_lane_wait_times()
        queue_lengths = self._get_lane_queue_lengths()

        total_wait = float(np.sum(wait_times))
        gini = self._calculate_gini(wait_times)
        reward = -(total_wait + self.gini_penalty_weight * gini)

        # ── Condiciones de fin de episodio ────────────────────────────────
        sim_time = self.traci_conn.simulation.getTime()
        n_loaded = self.traci_conn.simulation.getMinExpectedNumber()
        terminated = (sim_time >= _MAX_SIM_SECONDS) or (n_loaded == 0)
        truncated = False

        obs = self._get_observation()
        info: dict = {
            "sim_time_s": float(sim_time),
            "total_wait_s": total_wait,
            "gini_index": gini,
            "queue_total": float(np.sum(queue_lengths)),
            "current_phase": self._current_phase,
            "phase_age_s": self._phase_age,
            "reward_components": {
                "efficiency": -total_wait,
                "equity_penalty": -(self.gini_penalty_weight * gini),
            },
        }
        return obs, reward, terminated, truncated, info

    # =========================================================================
    # CLOSE
    # =========================================================================

    def close(self) -> None:
        """Cierra la conexión TraCI y el proceso SUMO de forma segura."""
        self._close_traci()
        # Limpiar archivos temporales
        try:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass
        logger.info("TrafficEnv: entorno cerrado correctamente.")

    # =========================================================================
    # RENDER (optional)
    # =========================================================================

    def render(self) -> None:
        """No-op: usar use_gui=True para visualización en tiempo real."""
        pass

    # =========================================================================
    # MÉTODOS PRIVADOS — TraCI lifecycle
    # =========================================================================

    def _start_traci(self, cfg_path: Path) -> None:
        """
        Inicia SUMO y establece una conexión TraCI completamente aislada.

        Estrategia de puerto dinámico:
            1. ``sumolib.miscutils.getFreeSocketPort()`` reserva atómicamente
               un puerto TCP libre a nivel del SO.
            2. ``traci.start(sumo_cmd, port=free_port, label=self.label)``
               lanza el proceso SUMO y conecta TraCI en ese puerto único.
            3. No se usa ``--remote-port`` en el comando: TraCI inyecta el
               argumento internamente al llamar a ``traci.start()``.

        Este mecanismo elimina la condición de carrera entre los workers de
        SubprocVecEnv que causaba el WinError 10061 (puerto ocupado).

        Args:
            cfg_path: Ruta al .sumocfg del episodio actual.
        """
        if not _TRACI_OK:
            raise ImportError(
                "traci / sumolib no están disponibles.\n"
                "Instala con: pip install traci==1.26.0 sumolib==1.26.0"
            )

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        # Reservar un puerto libre de forma atómica (evita colisiones en paralelo)
        free_port = sumolib.miscutils.getFreeSocketPort()
        self._traci_port = free_port

        # Comando SUMO SIN --remote-port: traci.start() lo inyecta internamente
        sumo_cmd = [
            sumo_binary,
            "-c", str(cfg_path),
            "--num-clients", "1",
            "-W",                        # silenciar warnings nativos
            "--no-step-log",             # silenciar log de pasos
            "--collision.action", "warn",
        ]

        logger.debug(
            f"  [{self.label[:12]}] traci.start() en puerto {free_port}"
        )
        try:
            # traci.start() lanza SUMO, conecta TraCI y registra la etiqueta
            traci.start(sumo_cmd, port=free_port, label=self.label)
            self.traci_conn = traci.getConnection(self.label)
            self._connected = True
            # _sumo_proc ya no es necesario: traci.start gestiona el proceso
            self._sumo_proc = None
            logger.debug(f"  [{self.label[:12]}] TraCI conectado OK.")
        except Exception as exc:
            raise ConnectionError(
                f"[{self.label[:12]}] No se pudo iniciar SUMO+TraCI "
                f"en puerto {free_port}: {exc}"
            ) from exc

    def _close_traci(self) -> None:
        """
        Cierra la conexión TraCI etiquetada y el proceso SUMO.

        Incluye un ``time.sleep(1.5)`` para que Windows libere el puerto TCP
        del socket en estado TIME_WAIT antes de que el siguiente reset()
        intente abrir otra instancia de SUMO en el mismo rango de puertos.
        """
        if self._connected and self.traci_conn is not None:
            try:
                self.traci_conn.close()
            except Exception:
                pass
            self.traci_conn = None
            self._connected = False

        if self._sumo_proc is not None:
            try:
                self._sumo_proc.terminate()
                self._sumo_proc.wait(timeout=5)
            except Exception:
                try:
                    self._sumo_proc.kill()
                except Exception:
                    pass
            self._sumo_proc = None

        # Pausa para liberar el puerto TCP en Windows (estado TIME_WAIT)
        time.sleep(1.5)

    # =========================================================================
    # MÉTODOS PRIVADOS — Intersección
    # =========================================================================

    def _init_intersection(self) -> None:
        """
        Inicializa los metadatos de la intersección controlada:
            - Carriles entrantes controlados por el semáforo.
            - Índices de fases verdes disponibles.
            - Espacios de observación y acción con las dimensiones reales.
        """
        # Carriles controlados — usando conexión explícita self.traci_conn
        try:
            self._controlled_lanes = list(
                self.traci_conn.trafficlight.getControlledLanes(self.tls_id)
            )
            # Eliminar duplicados preservando orden
            seen: set[str] = set()
            unique_lanes: list[str] = []
            for lane in self._controlled_lanes:
                if lane not in seen:
                    seen.add(lane)
                    unique_lanes.append(lane)
            self._controlled_lanes = unique_lanes
        except traci.exceptions.TraCIException:
            logger.warning(
                f"  tls_id='{self.tls_id}' no encontrado. "
                f"Semáforos disponibles: {self.traci_conn.trafficlight.getIDList()}"
            )
            self._controlled_lanes = []

        # Fases del semáforo
        try:
            logic = self.traci_conn.trafficlight.getAllProgramLogics(self.tls_id)[0]
            all_phases = logic.phases
            self._n_phases = len(all_phases)
            self._green_phases = [
                i for i, ph in enumerate(all_phases)
                if _is_green_phase(ph.state)
            ]
        except (traci.exceptions.TraCIException, IndexError):
            self._n_phases = 4
            self._green_phases = [0, 2]   # fallback conservador

        if not self._green_phases:
            self._green_phases = list(range(self._n_phases))

        n_lanes = len(self._controlled_lanes)
        n_green = len(self._green_phases)

        # NO redefinir observation_space ni action_space aquí.
        # SB3 ya los usó para construir la red en __init__.
        # _get_observation() padea con ceros hasta self._obs_dim_fixed.
        # step() clipea action a min(action, len(green_phases)-1).
        logger.info(
            f"  Intersección '{self.tls_id}': "
            f"{n_lanes} carriles controlados, {n_green} fases verdes "
            f"(obs fija={self._obs_dim_fixed}, actions fijas={self._n_green_phases_fixed})"
        )

    # =========================================================================
    # MÉTODOS PRIVADOS — Observación y recompensa
    # =========================================================================

    def _get_lane_wait_times(self) -> np.ndarray:
        """
        Retorna un array con el tiempo de espera acumulado (s) por carril.

        Usa traci.lane.getWaitingTime() que devuelve la suma de los tiempos
        de espera de todos los vehículos detenidos en el carril.
        """
        waits = np.zeros(len(self._controlled_lanes), dtype=np.float64)
        for i, lane in enumerate(self._controlled_lanes):
            try:
                waits[i] = self.traci_conn.lane.getWaitingTime(lane)
            except traci.exceptions.TraCIException:
                pass
        return waits

    def _get_lane_queue_lengths(self) -> np.ndarray:
        """
        Retorna un array con el número de vehículos detenidos por carril.

        Usa traci.lane.getLastStepHaltingNumber() — vehículos con
        velocidad < 0.1 m/s en el último paso de simulación.
        """
        queues = np.zeros(len(self._controlled_lanes), dtype=np.float64)
        for i, lane in enumerate(self._controlled_lanes):
            try:
                queues[i] = self.traci_conn.lane.getLastStepHaltingNumber(lane)
            except traci.exceptions.TraCIException:
                pass
        return queues

    def _get_observation(self) -> np.ndarray:
        """
        Construye el vector de observación normalizado al rango [0, 1] con
        tamaño FIJO igual a ``self._obs_dim_fixed`` (= 34 por defecto).

        Estructura:
            obs[:N]     = queue por carril normalizado  (N = n_lanes real)
            obs[N:2N]   = wait  por carril normalizado
            obs[-2]     = fase activa normalizada ∈ [0,1]
            obs[-1]     = edad de la fase normalizada ∈ [0,1]

        Si ``n_lanes < _N_LANES_MAX``, los slots sobrantes se rellenan con 0.
        Esto garantiza que el shape coincida siempre con ``observation_space``.

        Returns:
            Array float32 plano de shape ``(self._obs_dim_fixed,)``.
        """
        # Retornar vector nulo si TraCI no está activo
        if not self._connected or not self._controlled_lanes:
            return np.zeros(self._obs_dim_fixed, dtype=np.float32)

        queues_raw = self._get_lane_queue_lengths()   # shape (n_lanes,)
        waits_raw  = self._get_lane_wait_times()      # shape (n_lanes,)

        n_lanes = len(self._controlled_lanes)
        n_lanes_max = (self._obs_dim_fixed - 2) // 2  # = _N_LANES_MAX = 16

        # Normalizar y truncar/padear a n_lanes_max elementos
        q_norm = np.zeros(n_lanes_max, dtype=np.float32)
        w_norm = np.zeros(n_lanes_max, dtype=np.float32)
        n_use = min(n_lanes, n_lanes_max)

        q_norm[:n_use] = np.clip(queues_raw[:n_use] / _MAX_QUEUE, 0.0, 1.0)
        w_norm[:n_use] = np.clip(waits_raw[:n_use]  / _MAX_WAIT,  0.0, 1.0)

        # Escalar fase activa y edad
        n_green = max(len(self._green_phases), 1)
        phase_norm = (
            self._green_phases.index(self._current_phase) / n_green
            if self._current_phase in self._green_phases
            else 0.0
        )
        age_norm = float(min(self._phase_age / _MAX_PHASE_AGE, 1.0))

        # Concatenar en un vector plano 1-D de tamaño fijo
        obs = np.concatenate([q_norm, w_norm, [phase_norm, age_norm]])
        assert obs.shape == (self._obs_dim_fixed,), (
            f"Shape mismatch: obs={obs.shape} vs fixed={self._obs_dim_fixed}"
        )
        return obs.astype(np.float32)

    @staticmethod
    def _calculate_gini(wait_times: np.ndarray) -> float:
        """
        Calcula el Índice de Gini espacial de los tiempos de espera entre carriles.

        Mide la inequidad en la distribución del tiempo de espera:
            G = 0  →  todos los carriles esperan lo mismo (equidad perfecta)
            G = 1  →  toda la espera se concentra en un único carril

        Fórmula vectorizada O(n log n):
            G = (2 · Σ i·xᵢ) / (n · Σ xᵢ) − (n+1)/n
        donde xᵢ son los tiempos de espera ordenados ascendentemente.

        Args:
            wait_times: Array de tiempos de espera por carril (≥ 0).

        Returns:
            Índice de Gini en [0, 1].
        """
        arr = np.asarray(wait_times, dtype=np.float64)
        if arr.size == 0 or arr.sum() == 0:
            return 0.0

        arr = np.sort(arr)                      # orden ascendente
        n = arr.size
        idx = np.arange(1, n + 1, dtype=np.float64)
        gini = (2.0 * np.dot(idx, arr)) / (n * arr.sum()) - (n + 1.0) / n
        return float(np.clip(gini, 0.0, 1.0))
