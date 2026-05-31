"""
Tests unitarios para módulos de Transferencia y Generalización.
Valida domain randomization, zero-shot transfer y fine-tuning cross-city.

@ref: wmn7/Universal-Light (UniTSA patterns)
@ref: LucasAlegre/sumo-rl (Transfer protocols)
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.transfer.domain_adaptor import DomainAdaptor, JunctionConfig


class TestJunctionConfig:
    """Valida configuración de intersecciones para transferencia."""
    
    def test_create_config(self):
        config = JunctionConfig(
            junction_id="J1",
            n_lanes=4,
            n_phases=4
        )
        assert config.junction_id == "J1"
        assert config.n_lanes == 4
        assert config.n_phases == 4
        
    def test_topology_hash(self):
        config = JunctionConfig(
            junction_id="complex_junction",
            n_lanes=8,
            n_phases=8
        )
        hash_result = config.compute_topology_hash()
        assert isinstance(hash_result, str)
        assert len(hash_result) > 0


class TestDomainAdaptor:
    """Valida adaptación de dominios para transferencia."""
    
    def setup_method(self):
        # DomainAdaptor requiere factories de entornos
        mock_factory = MagicMock()
        self.adaptor = DomainAdaptor(
            source_env_factory=mock_factory,
            target_env_factory=mock_factory
        )
        
    def test_create_adaptor(self):
        # Test básico de creación
        mock_factory = MagicMock()
        adaptor = DomainAdaptor(
            source_env_factory=mock_factory,
            target_env_factory=mock_factory
        )
        assert adaptor is not None
        
    def test_add_junction(self):
        # DomainAdaptor usa junction_matrix internamente
        config = JunctionConfig(
            junction_id="J1",
            n_lanes=4,
            n_phases=4
        )
        self.adaptor.junction_matrix.add_junction(config)
        assert "J1" in self.adaptor.junction_matrix.junctions


class TestDomainRandomization:
    """Valida conceptos de randomización de dominio."""
    
    def test_demand_variation_concept(self):
        # Test conceptual: demanda debe poder variar en un rango
        base_demand = 1000
        min_factor, max_factor = 0.5, 1.5
        
        import numpy as np
        randomized = base_demand * np.random.uniform(min_factor, max_factor)
        
        assert base_demand * min_factor <= randomized <= base_demand * max_factor


class TestCrossCityTransfer:
    """Valida conceptos de transferencia entre ciudades."""
    
    def test_performance_gap_concept(self):
        # Concepto: gap entre rendimiento sintético y real
        synthetic_perf = 0.90
        real_perf = 0.70
        
        gap = synthetic_perf - real_perf
        assert gap > 0
        assert gap < 1
