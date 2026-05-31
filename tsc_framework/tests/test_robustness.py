"""
Tests unitarios para módulos de Robustez (CAREL/ERNIE patterns).
Valida inyección de ruido, fallos de sensores y defensas adversariales.

@ref: fmpr/CAREL (Robustness under uncertainty)
@ref: abukharin3/ERNIE (Adversarial regularization)
"""
import pytest
import numpy as np
from src.robustness.stress_injector import PerturbationConfig, StressInjector, PerturbationType
from src.robustness.metrics import RobustnessMetrics


class TestPerturbationConfig:
    """Valida la configuración de perturbaciones."""
    
    def test_default_config(self):
        config = PerturbationConfig(perturbation_type=PerturbationType.SENSOR_NOISE)
        assert config.severity == 0.1
        assert config.probability == 0.2
        assert config.duration_steps == 10
        
    def test_custom_config(self):
        config = PerturbationConfig(
            perturbation_type=PerturbationType.DETECTION_FAILURE,
            severity=0.3,
            probability=0.5,
            duration_steps=20
        )
        assert config.severity == 0.3
        assert config.probability == 0.5
        assert config.duration_steps == 20
        
    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            PerturbationConfig(perturbation_type=PerturbationType.SENSOR_NOISE, severity=1.5)  # > 1.0
        with pytest.raises(ValueError):
            PerturbationConfig(perturbation_type=PerturbationType.SENSOR_NOISE, probability=-0.1)  # < 0


class TestStressInjector:
    """Valida la inyección de estrés en observaciones."""
    
    def setup_method(self):
        self.injector = StressInjector(
            perturbation_type=PerturbationType.SENSOR_NOISE,
            severity=0.1,
            probability=1.0,  # Siempre activo para tests
            seed=42
        )
        self.base_obs = np.ones(34) * 0.5  # Observación base normalizada
        
    def test_no_perturbation(self):
        # Injector con probabilidad 0 nunca se activa
        injector_inactive = StressInjector(
            perturbation_type=PerturbationType.SENSOR_NOISE,
            probability=0.0,
            seed=42
        )
        obs = injector_inactive.inject(self.base_obs)
        np.testing.assert_array_equal(obs, self.base_obs)
        
    def test_gaussian_noise(self):
        injector_noise = StressInjector(
            perturbation_type=PerturbationType.SENSOR_NOISE,
            severity=0.3,
            probability=1.0,
            seed=42
        )
        obs = injector_noise.inject(self.base_obs)
        # Debe tener ruido pero mantenerse cerca del original
        assert not np.array_equal(obs, self.base_obs)
        assert np.allclose(obs, self.base_obs, atol=0.5)  # Tolerancia amplia
        # Debe estar clippeado a [0, 1]
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        
    def test_detection_failure(self):
        injector_fail = StressInjector(
            perturbation_type=PerturbationType.DETECTION_FAILURE,
            severity=0.5,
            probability=1.0,
            seed=42
        )
        obs = injector_fail.inject(self.base_obs)
        # Al menos algunos valores deberían ser cero
        assert np.any(obs == 0)
        
    def test_deterministic_seed(self):
        config = PerturbationConfig(perturbation_type=PerturbationType.SENSOR_NOISE, severity=0.1)
        injector1 = StressInjector(severity=0.1, probability=1.0, seed=42)
        injector2 = StressInjector(severity=0.1, probability=1.0, seed=42)
        
        obs1 = injector1.inject(self.base_obs)
        obs2 = injector2.inject(self.base_obs)
        np.testing.assert_array_equal(obs1, obs2)


class TestRobustnessMetrics:
    """Valida métricas de robustez (Performance Drop, Recovery Time)."""
    
    def test_performance_drop(self):
        baseline = 100.0
        stressed = 80.0
        drop = RobustnessMetrics.performance_drop(baseline, stressed)
        assert drop == 20.0  # 20% de caída
        
    def test_recovery_time(self):
        # Simular episodios: [100, 80, 90, 95, 99, 100]
        rewards = [100, 80, 90, 95, 99, 100]
        baseline = 100
        threshold = 0.95  # 95% de recuperación
        time = RobustnessMetrics.recovery_time(rewards, baseline, threshold)
        assert time == 0  # Índice 0 (valor 100) ya es >= 95 (threshold * baseline)
        
    def test_stability_index(self):
        # Baja varianza = alta estabilidad
        stable_rewards = [100, 101, 99, 100, 100]
        unstable_rewards = [100, 50, 150, 20, 180]
        
        stab_stable = RobustnessMetrics.stability_index(stable_rewards)
        stab_unstable = RobustnessMetrics.stability_index(unstable_rewards)
        
        assert stab_stable > stab_unstable
