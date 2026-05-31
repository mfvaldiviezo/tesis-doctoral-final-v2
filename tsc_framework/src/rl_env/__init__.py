# src/rl_env/__init__.py
"""
Entorno de Aprendizaje por Refuerzo (Gymnasium + SUMO TraCI)
=============================================================

NOTA DE REFACTORIZACIÓN:
Este módulo ha sido consolidado en src/core/tsc_env.py (TSCEnv).
Las clases aquí definidas son wrappers de compatibilidad hacia atrás.

Para nuevo desarrollo, usar directamente:
    from src.core.tsc_env import TSCEnv

Referencias Académicas:
    • Cap 4.2.2: Formulación del espacio de estados (34D)
    • Cap 4.3.2: Función de recompensa multiobjetivo
"""

# Importación de compatibilidad - redirige a implementación unificada
from src.core.tsc_env import TSCEnv as SumoRLEnv
from src.core.tsc_env import TSCConfig, DEFAULT_CONFIG

__all__ = ["SumoRLEnv", "TSCConfig", "DEFAULT_CONFIG"]
