"""
test_sumo_env.py — Tests unitarios para TSCEnv/SumoRLEnv (sin SUMO instalado)
==============================================================================
Usa mocks de TraCI para verificar la lógica del entorno en aislamiento.
Ejecutar con:  pytest tests/test_sumo_env.py -v

NOTA DE REFACTORIZACIÓN:
Los tests ahora apuntan a src/core/tsc_env.py (TSCEnv), que es la implementación
unificada. SumoRLEnv en rl_env es un alias de compatibilidad hacia atrás.
"""

from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ── Fixture: mock completo de traci ──────────────────────────────────────────

def _build_traci_mock() -> MagicMock:
    """Construye un mock de traci con los métodos usados en TSCEnv."""
    traci_mock = MagicMock()

    # trafficlight - 12 carriles controlados según Cap 4.2.2
    traci_mock.trafficlight.getControlledLanes.return_value = [
        f"lane_{i}" for i in range(12)
    ]
    
    # Mock para getAllProgramLogics (API SUMO 1.26.0)
    logic_mock = MagicMock()
    logic_mock.phases = [
        MagicMock(state="GGrr"),   # fase 0: verde
        MagicMock(state="yyrr"),   # fase 1: amarilla
        MagicMock(state="rrGG"),   # fase 2: verde
        MagicMock(state="rryy"),   # fase 3: amarilla
    ]
    traci_mock.trafficlight.getAllProgramLogics.return_value = [logic_mock]
    
    # Fallback para getCompleteRedYellowGreenDefinition (versiones antiguas)
    traci_mock.trafficlight.getCompleteRedYellowGreenDefinition.return_value = [logic_mock]
    
    traci_mock.trafficlight.getPhase.return_value = 0

    # lane - datos por defecto (se puede sobreescribir con side_effect)
    traci_mock.lane.getLastStepHaltingNumber.return_value = 3
    traci_mock.lane.getWaitingTime.return_value = 12.5
    traci_mock.lane.getLastStepVehicleNumber.return_value = 5

    # simulation
    traci_mock.simulation.getTime.return_value = 100.0
    traci_mock.simulation.getEndTime.return_value = 3600.0

    return traci_mock


@pytest.fixture
def mock_traci(monkeypatch):
    """Inyecta el mock de traci en el módulo core.tsc_env."""
    traci_mock = _build_traci_mock()
    # Parchear a nivel de módulo para que tsc_env.py lo vea
    import src.core.tsc_env as env_mod
    monkeypatch.setattr(env_mod, "traci", traci_mock)
    monkeypatch.setattr(env_mod, "TRACI_AVAILABLE", True)
    return traci_mock


@pytest.fixture
def env(tmp_path, mock_traci):
    """Crea una instancia de TSCEnv con una .sumocfg falsa."""
    from src.core.tsc_env import TSCEnv

    # Crear archivo .sumocfg falso para que Path.exists() pase
    cfg = tmp_path / "test.sumocfg"
    cfg.write_text("<configuration/>")

    instance = TSCEnv(
        sumocfg_path=cfg,
        tls_id="J0",
        use_gui=False,
        delta_t=5,
        max_steps=100,
        seed=42,
    )
    # Parchear _launch_sumo para no lanzar proceso real
    instance._launch_sumo = MagicMock()
    return instance


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSumoRLEnvReset:

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert "tls_id" in info
        assert "n_controlled_lanes" in info or "n_lanes" in info

    def test_obs_shape_matches_observation_space(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_obs_values_in_unit_range(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= 0.0), "Hay valores negativos en la observación"
        assert np.all(obs <= 1.0), "Hay valores > 1 en la observación"

    def test_obs_dtype_is_float32(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_n_green_phases_detected(self, env):
        env.reset()
        # El mock define fases 0 y 2 como verdes (contienen 'G')
        assert len(env._green_phases) == 2

    def test_action_space_matches_green_phases(self, env):
        env.reset()
        assert env.action_space.n == 4  # Discrete(4) según config

    def test_obs_dim_formula(self, env):
        env.reset()
        # Según Cap 4.2.2: q(12) + w(12) + p(4) + φ(4) + τ(2) = 34
        assert env._obs_dim == 34


class TestSumoRLEnvStep:

    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_step_obs_in_valid_range(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_step_reward_is_non_positive(self, env):
        """La recompensa base siempre es ≤ 0 (sólo penalizaciones)."""
        env.reset()
        _, reward, *_ = env.step(0)
        assert reward <= 0.0

    def test_step_info_has_required_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        for key in ("step", "sim_time", "phase", "phase_age", "total_queue"):
            assert key in info, f"Falta la clave '{key}' en info"

    def test_step_increments_counter(self, env):
        env.reset()
        env.step(0)
        assert env._step_count == 1
        env.step(0)
        assert env._step_count == 2

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.step(999)  # Fuera del rango de fases verdes


class TestRewardComponents:

    def test_delay_is_non_negative(self, env):
        env.reset()
        delay = env._compute_delay()
        assert delay >= 0.0

    def test_delay_equals_sum_of_wait_times(self, env, mock_traci):
        """_compute_delay debe ser igual a n_lanes * mock_wait_time."""
        env.reset()
        # El mock retorna 12.5 por carril y hay 12 carriles controlados
        assert env._compute_delay() == pytest.approx(12 * 12.5)

    def test_pressure_vector_is_non_negative(self, env):
        """_get_pressure debe retornar vector con valores no negativos."""
        env.reset()
        pressure = env._get_pressure()
        assert np.all(pressure >= 0.0)

    def test_reward_weights_sum_to_one(self, env):
        total = sum(env.reward_weights.values())
        assert abs(total - 1.0) < 1e-6, f"Pesos no suman 1: {total}"

    def test_reward_uses_real_gini_not_zero(self, env, mock_traci):
        """Con tiempos de espera distintos, Gini debe ser > 0."""
        env.reset()
        # Simular distribución desigual: 12 carriles con valores variados
        wait_values = [0.0, 0.0, 0.0, 100.0] * 3  # 12 valores para 12 carriles
        mock_traci.lane.getWaitingTime.side_effect = wait_values
        wait = env._get_wait_times()
        gini = env._calculate_gini(wait)
        assert gini > 0.0

    def test_reward_uses_real_cvar_not_zero(self, env, mock_traci):
        """Con tiempos de espera positivos, CVaR90 debe ser > 0."""
        env.reset()
        wait = np.array([10.0, 20.0, 30.0, 100.0], dtype=np.float32)
        cvar = env._calculate_cvar(wait, alpha=0.90)
        assert cvar > 0.0


class TestGiniAndCVaR:
    """Tests de propiedades matemáticas de _calculate_gini y _calculate_cvar."""

    # ── Gini ─────────────────────────────────────────────────────────────────

    def test_gini_perfect_equality(self, env):
        """Todos los valores iguales → G = 0."""
        env.reset()
        w = np.array([10.0, 10.0, 10.0, 10.0])
        assert env._calculate_gini(w) == pytest.approx(0.0, abs=1e-9)

    def test_gini_max_inequality(self, env):
        """Un solo carril con toda la espera → G debe ser cercano a 1."""
        env.reset()
        w = np.array([0.0, 0.0, 0.0, 100.0])
        g = env._calculate_gini(w)
        # Con n=4 y concentración máxima, G = (2·4·100)/(4·100) - 5/4 = 2 - 1.25 = 0.75
        assert g == pytest.approx(0.75, abs=1e-9)

    def test_gini_all_zeros_returns_zero(self, env):
        """Sin espera → G = 0 (sin división por cero)."""
        env.reset()
        w = np.zeros(4)
        assert env._calculate_gini(w) == 0.0

    def test_gini_empty_array_returns_zero(self, env):
        env.reset()
        assert env._calculate_gini(np.array([])) == 0.0

    def test_gini_in_unit_range(self, env):
        """G siempre debe estar en [0, 1]."""
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(50):
            w = rng.uniform(0, 300, size=rng.integers(2, 12)).astype(np.float32)
            g = env._calculate_gini(w)
            assert 0.0 <= g <= 1.0, f"Gini={g} fuera de [0,1] para w={w}"

    def test_gini_symmetric(self, env):
        """Gini debe ser invariante al orden de los carriles."""
        env.reset()
        w = np.array([5.0, 20.0, 50.0, 3.0])
        assert env._calculate_gini(w) == pytest.approx(
            env._calculate_gini(w[::-1]), abs=1e-9
        )

    # ── CVaR ─────────────────────────────────────────────────────────────────

    def test_cvar_equals_max_for_alpha_near_one(self, env):
        """CVaR al 99% sobre 4 elementos ≈ el valor máximo."""
        env.reset()
        w = np.array([1.0, 2.0, 3.0, 100.0])
        cvar = env._calculate_cvar(w, alpha=0.99)
        assert cvar == pytest.approx(100.0, abs=1e-6)

    def test_cvar_equals_mean_for_alpha_zero(self, env):
        """CVaR al 0% es la media completa (todos en la cola)."""
        env.reset()
        w = np.array([10.0, 20.0, 30.0, 40.0])
        cvar = env._calculate_cvar(w, alpha=0.0)
        assert cvar == pytest.approx(float(w.mean()), abs=1e-6)

    def test_cvar_empty_returns_zero(self, env):
        env.reset()
        assert env._calculate_cvar(np.array([]), alpha=0.90) == 0.0

    def test_cvar_non_negative(self, env):
        """CVaR siempre debe ser >= 0."""
        env.reset()
        rng = np.random.default_rng(1)
        for _ in range(50):
            w = rng.uniform(0, 300, size=rng.integers(2, 12)).astype(np.float32)
            assert env._calculate_cvar(w, alpha=0.90) >= 0.0

    def test_cvar_geq_mean(self, env):
        """CVaR_α >= media porque sólo promedia la cola superior."""
        env.reset()
        w = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        cvar = env._calculate_cvar(w, alpha=0.80)
        assert cvar >= float(w.mean()) - 1e-9


class TestStateComponents:

    def test_queue_lengths_shape(self, env):
        env.reset()
        q = env._get_queue_lengths()
        assert q.shape == (env._n_lanes,)

    def test_wait_times_shape(self, env):
        env.reset()
        w = env._get_wait_times()
        assert w.shape == (env._n_lanes,)

    def test_pressure_shape(self, env):
        env.reset()
        p = env._get_pressure()
        assert p.shape == (env._n_lanes,)


class TestEnvClose:

    def test_close_does_not_raise(self, env):
        env.reset()
        env.close()  # No debe lanzar excepción

    def test_repr_contains_key_info(self, env):
        env.reset()
        r = repr(env)
        assert "TSCEnv" in r
        assert "J0" in r
