# src/core/__init__.py
"""
Módulo Core del Framework TSC
==============================
Contiene la implementación unificada del entorno RL y componentes centrales.

Referencias Académicas:
    • Cap 4.2.2: Formulación del espacio de estados (34D)
    • Cap 4.3.2: Función de recompensa multiobjetivo
    • Apéndice A.4: Hiperparámetros de PPO

Componentes:
    TSCEnv        - Entorno Gymnasium unificado para control semafórico
    TSCConfig     - Configuración inmutable del dominio
    DEFAULT_CONFIG - Instancia de configuración por defecto
"""

from .tsc_env import TSCEnv, TSCConfig, DEFAULT_CONFIG

__all__ = ["TSCEnv", "TSCConfig", "DEFAULT_CONFIG"]
