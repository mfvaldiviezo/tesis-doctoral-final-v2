"""
Callbacks personalizados para entrenamiento de agentes RL.

Implementación de callbacks para registro de métricas de riesgo y equidad
en TensorBoard, siguiendo la formulación del Capítulo 4.3.2 de la tesis.

Referencias:
    - Capítulo 4.3.2: Función de recompensa multiobjetivo
    - Apéndice A.4: Protocolo de evaluación y registro de métricas
"""

import numpy as np
from typing import Any, Dict, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class RiskMetricsCallback(BaseCallback):
    """
    Callback personalizado para registrar métricas de riesgo y equidad en TensorBoard.
    
    Registra las siguientes métricas adicionales durante el entrenamiento:
    - delay_mean: Tiempo medio de espera por vehículo
    - gini: Coeficiente de Gini de la distribución de esperas
    - cvar_alpha: CVaR (Conditional Value at Risk) de las pérdidas
    
    Esta implementación sigue estrictamente la formulación matemática del
    Capítulo 4.3.2 de la tesis doctoral.
    
    Args:
        verbose: Nivel de verbosidad (0 = silencioso, 1 = info, 2 = debug)
        log_freq: Frecuencia de registro en steps
        alpha: Nivel de confianza para CVaR (por defecto 0.95)
    """
    
    def __init__(
        self,
        verbose: int = 0,
        log_freq: int = 1000,
        alpha: float = 0.95
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.alpha = alpha
        self.episode_losses = []
        self.episode_delays = []
        self.episode_ginis = []
        
    def _on_step(self) -> bool:
        """
        Ejecutado después de cada step de entrenamiento.
        
        Returns:
            True para continuar el entrenamiento, False para detenerlo.
        """
        # Registrar métricas si están disponibles en las infos
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                if isinstance(info, dict):
                    if 'delay' in info:
                        self.episode_delays.append(info['delay'])
                    if 'gini' in info:
                        self.episode_ginis.append(info['gini'])
                    if 'loss' in info:
                        self.episode_losses.append(info['loss'])
        
        # Loggear métricas acumuladas cada log_freq steps
        if self.n_calls % self.log_freq == 0 and len(self.episode_delays) > 0:
            self._log_metrics()
            
        return True
    
    def _log_metrics(self):
        """Registra las métricas acumuladas en TensorBoard."""
        if len(self.episode_delays) > 0:
            delay_mean = np.mean(self.episode_delays)
            self.logger.record('metrics/delay_mean', delay_mean)
            self.episode_delays = []
            
        if len(self.episode_ginis) > 0:
            gini_mean = np.mean(self.episode_ginis)
            self.logger.record('metrics/gini_mean', gini_mean)
            self.episode_ginis = []
            
        if len(self.episode_losses) > 0:
            losses = np.array(self.episode_losses)
            # Calcular VaR y CVaR
            var = np.percentile(losses, self.alpha * 100)
            cvar = np.mean(losses[losses >= var])
            self.logger.record('metrics/cvar_alpha', cvar)
            self.logger.record('metrics/var_alpha', var)
            self.episode_losses = []
    
    def _on_rollout_end(self) -> None:
        """Ejecutado al final de cada rollout."""
        # Loggear métricas de rollout si hay datos pendientes
        if len(self.episode_delays) > 0 or len(self.episode_ginis) > 0:
            self._log_metrics()


class EvaluationCallback(BaseCallback):
    """
    Callback para evaluación periódica del agente durante el entrenamiento.
    
    Evalúa el agente cada N steps y registra las métricas de rendimiento
    en TensorBoard. Permite detectar overfitting y seguir la evolución
    del rendimiento a lo largo del entrenamiento.
    
    Args:
        eval_env: Entorno de evaluación (puede ser el mismo o diferente)
        eval_freq: Frecuencia de evaluación en steps
        n_eval_episodes: Número de episodios para evaluación
        verbose: Nivel de verbosidad
    """
    
    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
    def _on_step(self) -> bool:
        """Evalúa el agente cada eval_freq steps."""
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/std_reward', std_reward)
            
            if self.verbose >= 1:
                print(f"Evaluación en step {self.n_calls}: "
                      f"reward={mean_reward:.2f} ± {std_reward:.2f}")
              
        return True
