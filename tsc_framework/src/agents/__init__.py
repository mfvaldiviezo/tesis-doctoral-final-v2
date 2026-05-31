# src/agents/__init__.py
"""
Módulo de Agentes RL
====================
Responsable de:
- Wrappers sobre stable-baselines3 (PPO, SAC, TD3)
- Integración de métricas de riesgo (CVaR, Gini) en la función de pérdida
- Callbacks de entrenamiento personalizados
- Guardado/carga de checkpoints con metadatos de configuración

Clases a implementar (Fase 3):
    PPOAgentWrapper   - PPO con penalización CVaR en la recompensa
    SACAgentWrapper   - SAC para espacios de acción continuos
    RiskCallback      - Callback SB3 para monitoreo de métricas de riesgo
    GiniPenalizer     - Penalizador de inequidad distributiva inter-carriles
"""
