# tsc_framework/src/__init__.py
"""
TSC Framework - Paquete Principal
==================================
Framework modular para control semafórico inteligente.

Módulos:
    data_pipeline  - Ingesta y limpieza de datos de tráfico
    copulas        - Generación probabilística de escenarios de estrés
    rl_env         - Entorno Gymnasium personalizado con TraCI/SUMO
    agents         - Wrappers de algoritmos RL (PPO/SAC) + métricas de riesgo
    utils          - Funciones auxiliares, logging y métricas
"""

__version__ = "0.1.0"
__author__ = "Doctoral Researcher"
