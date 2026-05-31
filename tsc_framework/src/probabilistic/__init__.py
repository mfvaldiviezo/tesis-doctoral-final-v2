"""
Módulo Probabilístico - Generación de Escenarios con Vine Copulas

Referencia Tesis Doctoral:
    Capítulo 4.3.1: "Generación de Escenarios de Tráfico mediante Vine Copulas"

Este módulo exporta las clases principales para generación de escenarios
de estrés basados en Regular Vine Copulas.

Clases exportadas:
    - VineCopulaGenerator: Pipeline completo de ajuste y muestreo
    - VineConfig: Configuración del vine
    - StressScenario: Escenario generado
    - MarginalFit: Ajuste de distribución marginal
    - MarginalDistribution: Tipos de distribuciones soportadas
    - generate_synthetic_data: Función utilitaria para datos de prueba
"""

from .vine_generator import (
    VineCopulaGenerator,
    VineConfig,
    StressScenario,
    MarginalFit,
    MarginalDistribution,
    generate_synthetic_data,
)

__all__ = [
    "VineCopulaGenerator",
    "VineConfig",
    "StressScenario",
    "MarginalFit",
    "MarginalDistribution",
    "generate_synthetic_data",
]
