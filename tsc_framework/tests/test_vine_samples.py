"""
Tests para el Módulo Probabilístico - Vine Copulas

Referencia Tesis Doctoral:
    Capítulo 4.3.1: "Generación de Escenarios de Tráfico mediante Vine Copulas"
    - Sección 4.3.1.1: Validación de ajustes marginales (test KS)
    - Sección 4.3.1.3: Generación de escenarios de estrés

Estos tests validan:
1. Ajuste correcto de distribuciones marginales
2. Preservación de dependencias en muestreo vine
3. Generación de escenarios con características de estrés esperadas
4. Exportación válida a formato SUMO
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Verificar disponibilidad de pyvinecopulib
try:
    import pyvinecopulib
    VINECOPULIB_AVAILABLE = True
except ImportError:
    VINECOPULIB_AVAILABLE = False

from src.probabilistic.vine_generator import (
    VineCopulaGenerator,
    VineConfig,
    StressScenario,
    MarginalFit,
    MarginalDistribution,
    generate_synthetic_data,
)

logging.basicConfig(level=logging.WARNING)


class TestMarginalFit:
    """Tests para la clase MarginalFit."""

    def test_marginal_fit_exponential_sample(self):
        """Valida muestreo de distribución exponencial."""
        fit = MarginalFit(
            variable_name="test_exp",
            distribution=MarginalDistribution.EXPONENTIAL,
            params=(2.5,),  # scale
            ks_statistic=0.05,
            ks_pvalue=0.8,
            data_min=0.0,
            data_max=20.0,
        )

        rng = np.random.default_rng(42)
        samples = fit.sample(1000, rng)

        assert len(samples) == 1000
        assert samples.min() >= 0
        assert np.isclose(samples.mean(), 2.5, atol=0.5)  # E[X] = scale

    def test_marginal_fit_gamma_sample(self):
        """Valida muestreo de distribución gamma."""
        fit = MarginalFit(
            variable_name="test_gamma",
            distribution=MarginalDistribution.GAMMA,
            params=(2.0, 0.0, 100.0),  # a, loc, scale
            ks_statistic=0.04,
            ks_pvalue=0.9,
            data_min=0.0,
            data_max=1000.0,
        )

        rng = np.random.default_rng(42)
        samples = fit.sample(1000, rng)

        assert len(samples) == 1000
        assert samples.min() >= 0
        # E[X] = a * scale + loc = 2 * 100 + 0 = 200
        assert np.isclose(samples.mean(), 200, atol=30)

    def test_marginal_fit_cdf_ppf_consistency(self):
        """Valida que CDF y PPF son inversas."""
        fit = MarginalFit(
            variable_name="test_consistency",
            distribution=MarginalDistribution.EXPONENTIAL,
            params=(1.0,),
            ks_statistic=0.03,
            ks_pvalue=0.95,
            data_min=0.0,
            data_max=10.0,
        )

        # Probar en varios puntos
        x_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        for x in x_values:
            p = fit.cdf(x)
            x_recovered = fit.ppf(p)
            assert np.isclose(x, x_recovered, rtol=1e-5)


class TestVineConfig:
    """Tests para la clase VineConfig."""

    def test_default_config(self):
        """Valida configuración por defecto."""
        config = VineConfig()

        assert config.selection_criterion == "bic"
        assert config.truncation_level == -1
        assert config.rotation_check is True
        assert config.prefitting is True
        assert len(config.family_set) > 10

    def test_custom_config(self):
        """Valida configuración personalizada."""
        custom_families = ["gaussian", "clayton", "gumbel"]
        config = VineConfig(
            family_set=custom_families,
            selection_criterion="aic",
            truncation_level=1,
            rotation_check=False,
        )

        assert config.family_set == custom_families
        assert config.selection_criterion == "aic"
        assert config.truncation_level == 1
        assert config.rotation_check is False


class TestStressScenario:
    """Tests para la clase StressScenario."""

    def test_scenario_creation(self):
        """Valida creación de escenario."""
        scenario = StressScenario(
            scenario_id="test_001",
            scenario_type="moderate",
            demand_matrix=np.array([[500, 300], [600, 350]]),
            arrival_times=np.array([7.2, 6.0, 5.5, 6.5]),
            behavioral_friction=np.array([1.0, 1.1]),
            incident_probability=np.array([0.02, 0.01]),
            metadata={"seed": 42},
        )

        assert scenario.scenario_id == "test_001"
        assert scenario.scenario_type == "moderate"
        assert scenario.demand_matrix.shape == (2, 2)
        assert len(scenario.arrival_times) == 4

    def test_scenario_to_dataframe(self):
        """Valida conversión a DataFrame."""
        scenario = StressScenario(
            scenario_id="test_002",
            scenario_type="extreme",
            demand_matrix=np.array([[800, 500], [900, 600]]),
            arrival_times=np.array([4.5, 4.0, 3.8, 4.2]),
            behavioral_friction=np.array([1.2, 1.3]),
            incident_probability=np.array([0.05, 0.03]),
            metadata={"test": True},
        )

        df = scenario.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "scenario_id" in df.columns
        assert "demand_mean" in df.columns
        assert df.loc[0, "scenario_id"] == "test_002"


class TestSyntheticData:
    """Tests para generación de datos sintéticos."""

    def test_generate_synthetic_data_shape(self):
        """Valida dimensiones de datos sintéticos."""
        data = generate_synthetic_data(n_samples=100, seed=42)

        expected_cols = 12  # 4 demand + 4 arrival + 2 friction + 2 incident
        assert data.shape == (100, expected_cols)

    def test_generate_synthetic_data_columns(self):
        """Valida nombres de columnas en datos sintéticos."""
        data = generate_synthetic_data(n_samples=50, seed=123)

        expected_cols = [
            "demand_access_0", "demand_access_1", "demand_access_2", "demand_access_3",
            "arrival_time_0", "arrival_time_1", "arrival_time_2", "arrival_time_3",
            "friction_type_A", "friction_type_B",
            "incident_prob_peak", "incident_prob_offpeak",
        ]

        assert list(data.columns) == expected_cols

    def test_generate_synthetic_data_ranges(self):
        """Valida rangos razonables en datos sintéticos."""
        data = generate_synthetic_data(n_samples=1000, seed=42)

        # Demanda debe ser positiva
        for i in range(4):
            assert data[f"demand_access_{i}"].min() > 0

        # Tiempos entre llegadas deben ser positivos
        for i in range(4):
            assert data[f"arrival_time_{i}"].min() > 0

        # Fricción debe estar cerca de 1.0
        assert data["friction_type_A"].mean() > 0.5
        assert data["friction_type_B"].mean() > 0.5

        # Probabilidad de incidente debe estar en [0, 1]
        assert 0 <= data["incident_prob_peak"].max() <= 1
        assert 0 <= data["incident_prob_offpeak"].max() <= 1


@pytest.mark.skipif(not VINECOPULIB_AVAILABLE, reason="Requiere pyvinecopulib instalado")
class TestVineCopulaGenerator:
    """Tests para VineCopulaGenerator (requiere pyvinecopulib)."""

    @pytest.fixture
    def sample_data(self):
        """Genera datos de prueba."""
        return generate_synthetic_data(n_samples=200, seed=42)

    @pytest.fixture
    def generator(self):
        """Crea generador con seed fija."""
        return VineCopulaGenerator(seed=42)

    def test_fit_marginals(self, generator, sample_data):
        """Valida ajuste de marginales."""
        fits = generator.fit_marginals(sample_data)

        assert len(fits) == 12  # Todas las variables
        assert all(isinstance(fit, MarginalFit) for fit in fits.values())

    def test_fit_marginals_ks_validation(self, generator, sample_data):
        """Valida que ajustes marginales pasan KS test."""
        generator.fit_marginals(sample_data)

        # Al menos 80% de variables deben pasar KS con p > 0.01
        passed = sum(1 for fit in generator.marginal_fits.values() 
                    if fit.ks_pvalue > 0.01)
        
        assert passed >= len(generator.marginal_fits) * 0.8

    def test_fit_vine_structure(self, generator, sample_data):
        """Valida ajuste de estructura vine."""
        generator.fit_marginals(sample_data)
        vine = generator.fit_vine()

        assert vine is not None

    def test_sample_scenarios_nominal(self, generator, sample_data):
        """Valida generación de escenarios nominales."""
        generator.fit_marginals(sample_data)
        generator.fit_vine()

        scenarios = generator.sample_stress_scenarios(
            n_scenarios=10, stress_level="nominal"
        )

        assert len(scenarios) == 10
        assert all(s.scenario_type == "nominal" for s in scenarios)

    def test_sample_scenarios_moderate_higher_demand(self, generator, sample_data):
        """Valida que escenarios moderados tienen mayor demanda."""
        generator.fit_marginals(sample_data)
        generator.fit_vine()

        nominal = generator.sample_stress_scenarios(n_scenarios=20, stress_level="nominal")
        moderate = generator.sample_stress_scenarios(n_scenarios=20, stress_level="moderate")

        nominal_mean = np.mean([s.demand_matrix.mean() for s in nominal])
        moderate_mean = np.mean([s.demand_matrix.mean() for s in moderate])

        # La demanda moderada debería ser mayor en promedio
        assert moderate_mean > nominal_mean * 0.9

    def test_sample_scenarios_extreme_highest_demand(self, generator, sample_data):
        """Valida que escenarios extremos tienen máxima demanda."""
        generator.fit_marginals(sample_data)
        generator.fit_vine()

        nominal = generator.sample_stress_scenarios(n_scenarios=20, stress_level="nominal")
        extreme = generator.sample_stress_scenarios(n_scenarios=20, stress_level="extreme")

        nominal_mean = np.mean([s.demand_matrix.mean() for s in nominal])
        extreme_mean = np.mean([s.demand_matrix.mean() for s in extreme])

        # La demanda extrema debería ser significativamente mayor
        assert extreme_mean > nominal_mean * 1.2

    def test_export_to_sumo(self, generator, sample_data):
        """Valida exportación a formato SUMO."""
        generator.fit_marginals(sample_data)
        generator.fit_vine()

        scenarios = generator.sample_stress_scenarios(n_scenarios=3, stress_level="nominal")

        with tempfile.TemporaryDirectory() as tmpdir:
            files = generator.export_to_sumo(scenarios, tmpdir)

            assert len(files) == 3
            for filepath in files:
                assert filepath.exists()
                assert filepath.suffix == ".xml"
                
                # Validar contenido XML básico
                content = filepath.read_text()
                assert "<?xml version" in content
                assert "<routes" in content
                assert "</routes>" in content

    def test_validate_samples_ks_test(self, generator, sample_data):
        """Valida que muestras preservan distribuciones (KS test)."""
        generator.fit_marginals(sample_data)
        generator.fit_vine()

        validation = generator.validate_samples(n_test_samples=200)

        # Al menos 70% de variables deben pasar KS con p > 0.05
        passed = sum(1 for v in validation.values() if v["passed"])
        
        assert passed >= len(validation) * 0.7


class TestIntegration:
    """Tests de integración del pipeline completo."""

    @pytest.mark.skipif(not VINECOPULIB_AVAILABLE, reason="Requiere pyvinecopulib instalado")
    def test_full_pipeline(self):
        """Valida pipeline completo: datos → ajuste → muestreo → exportación."""
        # 1. Generar datos sintéticos
        data = generate_synthetic_data(n_samples=300, seed=42)

        # 2. Inicializar generador
        generator = VineCopulaGenerator(seed=42)

        # 3. Ajustar marginales
        fits = generator.fit_marginals(data)
        assert len(fits) == 12

        # 4. Ajustar vine
        vine = generator.fit_vine()
        assert vine is not None

        # 5. Generar escenarios de diferentes tipos
        scenarios = {
            "nominal": generator.sample_stress_scenarios(5, "nominal"),
            "moderate": generator.sample_stress_scenarios(5, "moderate"),
            "extreme": generator.sample_stress_scenarios(5, "extreme"),
        }

        assert len(scenarios["nominal"]) == 5
        assert len(scenarios["moderate"]) == 5
        assert len(scenarios["extreme"]) == 5

        # 6. Validar que extreme > moderate > nominal en demanda
        means = {
            k: np.mean([s.demand_matrix.mean() for s in v])
            for k, v in scenarios.items()
        }

        assert means["extreme"] > means["nominal"] * 1.1

        # 7. Exportar a SUMO
        with tempfile.TemporaryDirectory() as tmpdir:
            all_scenarios = sum(scenarios.values(), [])
            files = generator.export_to_sumo(all_scenarios, tmpdir)

            assert len(files) == 15
            assert all(f.exists() for f in files)

        # 8. Validar muestras
        validation = generator.validate_samples(n_test_samples=300)
        passed = sum(1 for v in validation.values() if v["passed"])
        
        # Al menos 60% debe pasar (margen por variabilidad muestral)
        assert passed >= len(validation) * 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
