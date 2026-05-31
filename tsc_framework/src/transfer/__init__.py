"""
Transfer Module for TSC Framework.
===================================

Este módulo implementa mecanismos de transferencia y generalización cross-city,
inspirados en Universal-Light / UniTSA (SLR 2026, QA ≥ 3.0).

Componentes:
    - DomainAdaptor: Adaptación de dominio y fine-tuning cross-topology
    - ZeroShotEvaluator: Evaluación zero-shot en nuevas intersecciones
    - JunctionMatrix: Gestión de matrices de intersección para transferencia

Referencias SLR 2026:
    [1] Universal-Light / UniTSA - Transferencia y generalización cross-city
        URL: https://github.com/wmn7/Universal-Light
        Componente adaptado: domain_randomization y zero-shot_transfer

Autor: Doctoral Researcher
Versión: 1.0.0 (Integración SOTA SLR-2026)
"""

from .domain_adaptor import DomainAdaptor, ZeroShotEvaluator, JunctionMatrix

__all__ = [
    "DomainAdaptor",
    "ZeroShotEvaluator",
    "JunctionMatrix",
]
