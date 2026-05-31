"""
Módulo de Agentes de Aprendizaje por Refuerzo.

Implementación de agentes RL basados en PPO y SAC con métricas de riesgo
y equidad distributiva, siguiendo la formulación del Capítulo 4.3.2 de la tesis.

Referencias:
    - Capítulo 4.3.2: Función de recompensa multiobjetivo
    - Apéndice A.4: Hiperparámetros de entrenamiento PPO
"""

from rl_agent.ppo_agent import PPOAgent
from rl_agent.sac_agent import SACAgent
from rl_agent.callbacks import RiskMetricsCallback

__all__ = ['PPOAgent', 'SACAgent', 'RiskMetricsCallback']
