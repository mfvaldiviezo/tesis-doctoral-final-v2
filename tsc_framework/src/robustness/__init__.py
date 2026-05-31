"""
Robustness Module for TSC Framework.
=====================================

Este módulo implementa mecanismos de evaluación bajo incertidumbre,
inspirados en los frameworks CAREL y ERNIE (SLR 2026, QA ≥ 3.0).

Componentes:
    - StressInjector: Inyección de perturbaciones (ruido, fallos de detección, cambios de demanda)
    - PerturbationCallback: Callbacks para evaluación de robustez durante entrenamiento (requiere SB3)
    - AdversarialDefense: Regularización Lipschitz y defensa adversarial (requiere PyTorch)
    - RobustnessEvaluator: Evaluador sistemático de robustez multi-escenario

Referencias SLR 2026:
    [1] CAREL - Framework de evaluación bajo incertidumbre
        URL: https://github.com/fmpr/CAREL
        Componente extraído: Estructura de callbacks de perturbación
    
    [2] ERNIE - Regularización Lipschitz y entrenamiento adversarial
        URL: https://github.com/abukharin3/ERNIE
        Componente adaptado: Módulo de defensa a observaciones

Autor: Doctoral Researcher
Versión: 1.0.0 (Integración SOTA SLR-2026)
"""

from .stress_injector import StressInjector, PerturbationType, PerturbationConfig

# Importaciones condicionales para componentes que dependen de librerías externas
try:
    from .perturbation_callback import PerturbationCallback, RobustnessEvaluator, RobustnessMetrics
    PERTURBATION_CALLBACK_AVAILABLE = True
except ImportError:
    PerturbationCallback = None  # type: ignore
    RobustnessEvaluator = None  # type: ignore
    RobustnessMetrics = None  # type: ignore
    PERTURBATION_CALLBACK_AVAILABLE = False

try:
    from .adversarial_defense import AdversarialDefense, LipschitzRegularizer
    ADVERSARIAL_DEFENSE_AVAILABLE = True
except ImportError:
    AdversarialDefense = None  # type: ignore
    LipschitzRegularizer = None  # type: ignore
    ADVERSARIAL_DEFENSE_AVAILABLE = False

__all__ = [
    "StressInjector",
    "PerturbationType",
    "PerturbationConfig",
    "PERTURBATION_CALLBACK_AVAILABLE",
    "ADVERSARIAL_DEFENSE_AVAILABLE",
]

# Añadir componentes disponibles dinámicamente
if PERTURBATION_CALLBACK_AVAILABLE:
    __all__.extend(["PerturbationCallback", "RobustnessEvaluator", "RobustnessMetrics"])

if ADVERSARIAL_DEFENSE_AVAILABLE:
    __all__.extend(["AdversarialDefense", "LipschitzRegularizer"])
