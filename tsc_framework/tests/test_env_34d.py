"""
test_env_34d.py — Pruebas Unitarias para TSCEnv (Estado 34D)
=============================================================
Framework computacional para Tesis Doctoral.

Referencias Académicas:
    • Cap 4.2.2: Formulación del espacio de estados (34D)
    • Cap 4.3.2: Función de recompensa multiobjetivo

Estas pruebas validan la dimensionalidad, CPU-only y aislamiento TraCI
sin requerir SUMO instalado (usan mocks).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, PropertyMock


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_torch_cpu():
    """Mock de PyTorch que simula device CPU real."""
    class MockDevice:
        type = 'cpu'
    
    class MockTorch:
        @staticmethod
        def device(name='cpu'):
            return MockDevice()
        
        class cuda:
            @staticmethod
            def is_available():
                return False
    
    return MockTorch()


# ─────────────────────────────────────────────────────────────────────────────
# Tests de Dimensionalidad 34D
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationDimensionality:
    """Tests para validar el espacio de observación 34D."""
    
    def test_obs_dim_constant(self):
        """Verificar que la constante de dimensión es 34."""
        from src.core.tsc_env import DEFAULT_CONFIG
        
        obs_dim_expected = (
            DEFAULT_CONFIG.N_CONTROLLED_LANES +
            DEFAULT_CONFIG.N_CONTROLLED_LANES +
            4 +
            DEFAULT_CONFIG.N_GREEN_PHASES +
            2
        )
        assert obs_dim_expected == 34
    
    def test_observation_space_shape(self):
        """Verificar que observation_space tiene shape=(34,)."""
        with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
            with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                from src.core.tsc_env import TSCEnv
                
                env = TSCEnv(
                    sumocfg_path="/tmp/fake.sumocfg",
                    tls_id="J0",
                    seed=42
                )
                
                assert env.observation_space.shape == (34,)
                assert env.observation_space.dtype == np.float32
    
    def test_action_space_discrete_4(self):
        """Verificar que action_space es Discrete(4)."""
        with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
            with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                from src.core.tsc_env import TSCEnv
                from gymnasium import spaces
                
                env = TSCEnv(
                    sumocfg_path="/tmp/fake.sumocfg",
                    tls_id="J0",
                    seed=42
                )
                
                assert isinstance(env.action_space, spaces.Discrete)
                assert env.action_space.n == 4


# ─────────────────────────────────────────────────────────────────────────────
# Tests de CPU-Only
# ─────────────────────────────────────────────────────────────────────────────

class TestCPUOnlyEnforcement:
    """Tests para validar que el device es explícitamente CPU."""
    
    def test_device_is_cpu_with_real_torch(self):
        """Verificar que self.device.type == 'cpu' con torch real."""
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
        
        if has_torch:
            with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
                with patch('src.core.tsc_env.TORCH_AVAILABLE', True):
                    from src.core.tsc_env import TSCEnv
                    
                    env = TSCEnv(
                        sumocfg_path="/tmp/fake.sumocfg",
                        tls_id="J0",
                        seed=42
                    )
                    
                    assert hasattr(env, 'device')
                    assert env.device.type == 'cpu'
        else:
            # Si no hay torch, verificamos que device sea None
            with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
                with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                    from src.core.tsc_env import TSCEnv
                    
                    env = TSCEnv(
                        sumocfg_path="/tmp/fake.sumocfg",
                        tls_id="J0",
                        seed=42
                    )
                    
                    assert env.device is None
    
    def test_validate_cpu_device_method(self):
        """Verificar que validate_cpu_device() retorna True cuando TORCH_AVAILABLE=False."""
        with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
            with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                from src.core.tsc_env import TSCEnv
                
                env = TSCEnv(
                    sumocfg_path="/tmp/fake.sumocfg",
                    tls_id="J0",
                    seed=42
                )
                
                result = env.validate_cpu_device()
                # Cuando TORCH_AVAILABLE=False, validate_cpu_device retorna True por defecto
                assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests de Recompensa Multiobjetivo
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiobjectiveReward:
    """Tests para la función de recompensa R_t = -(λ1·Delay + λ2·Gini + λ3·CVaR)."""
    
    def test_reward_weights_sum_to_one(self):
        """Verificar que los pesos de recompensa suman 1."""
        with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
            with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                from src.core.tsc_env import TSCEnv
                
                env = TSCEnv(
                    sumocfg_path="/tmp/fake.sumocfg",
                    tls_id="J0",
                    seed=42
                )
                
                weight_sum = sum(env.reward_weights.values())
                assert np.isclose(weight_sum, 1.0)
    
    def test_gini_in_range(self):
        """Verificar que Gini ∈ [0, 1]."""
        with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
            with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                from src.core.tsc_env import TSCEnv
                
                env = TSCEnv(
                    sumocfg_path="/tmp/fake.sumocfg",
                    tls_id="J0",
                    seed=42
                )
                
                wait_times_cases = [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([10.0, 10.0, 10.0]),
                    np.array([0.0, 0.0, 30.0]),
                    np.array([5.0, 15.0, 25.0]),
                ]
                
                for wait_times in wait_times_cases:
                    gini = env._calculate_gini(wait_times)
                    assert 0.0 <= gini <= 1.0, f"Gini={gini} fuera de rango"
    
    def test_cvar_non_negative(self):
        """Verificar que CVaR ≥ 0."""
        with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
            with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                from src.core.tsc_env import TSCEnv
                
                env = TSCEnv(
                    sumocfg_path="/tmp/fake.sumocfg",
                    tls_id="J0",
                    seed=42
                )
                
                env._loss_history = [10.0, 20.0, 30.0, 40.0, 50.0]
                losses = np.array(env._loss_history)
                cvar = env._calculate_cvar(losses, alpha=0.95)
                assert cvar >= 0.0, f"CVaR={cvar} debe ser >= 0"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de Aislamiento TraCI
# ─────────────────────────────────────────────────────────────────────────────

class TestTraCIIsolation:
    """Tests para aislamiento de instancias en SubprocVecEnv."""
    
    def test_instance_uuid_unique(self):
        """Verificar que cada instancia tiene UUID único."""
        with patch('src.core.tsc_env.TRACI_AVAILABLE', True):
            with patch('src.core.tsc_env.TORCH_AVAILABLE', False):
                from src.core.tsc_env import TSCEnv
                
                env1 = TSCEnv(sumocfg_path="/tmp/fake.sumocfg", tls_id="J0", seed=42)
                env2 = TSCEnv(sumocfg_path="/tmp/fake.sumocfg", tls_id="J0", seed=42)
                
                assert env1._instance_uuid != env2._instance_uuid
    
    def test_find_free_port_returns_int(self):
        """Verificar que _find_free_port retorna un entero válido."""
        from src.core.tsc_env import _find_free_port
        
        port = _find_free_port(start=8813)
        assert isinstance(port, int)
        assert port >= 8813


# ─────────────────────────────────────────────────────────────────────────────
# Tests de Configuración
# ─────────────────────────────────────────────────────────────────────────────

class TestTSCConfig:
    """Tests para la configuración del dominio."""
    
    def test_config_constants(self):
        """Verificar constantes de configuración."""
        from src.core.tsc_env import DEFAULT_CONFIG
        
        assert DEFAULT_CONFIG.N_CONTROLLED_LANES == 12
        assert DEFAULT_CONFIG.N_GREEN_PHASES == 4
        assert DEFAULT_CONFIG.DELTA_T == 5
        assert DEFAULT_CONFIG.MAX_STEPS == 3600
        assert DEFAULT_CONFIG.CVAR_ALPHA == 0.95
    
    def test_config_immutable(self):
        """Verificar que TSCConfig es frozen (inmutable)."""
        from src.core.tsc_env import TSCConfig
        
        config = TSCConfig()
        with pytest.raises(Exception):
            config.MAX_QUEUE = 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
