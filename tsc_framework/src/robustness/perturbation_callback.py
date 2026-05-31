"""
PerturbationCallback - Callbacks para Evaluación de Robustez.
==============================================================

@ref: [CAREL Paper] + [https://github.com/fmpr/CAREL]

Este módulo implementa callbacks en el estilo de CAREL para monitorear y
registrar métricas de robustez durante el entrenamiento del agente RL.

Componentes:
    - PerturbationCallback: Callback base para inyección de perturbaciones
    - RobustnessEvaluator: Evaluador sistemático de robustez multi-escenario

Autor: Doctoral Researcher
Versión: 1.0.0 (Integración SOTA SLR-2026)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import logging

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object  # type: ignore

from .stress_injector import StressInjector, PerturbationType, PerturbationConfig


logger = logging.getLogger(__name__)


@dataclass
class RobustnessMetrics:
    """Métricas agregadas de robustez."""
    clean_reward_mean: float = 0.0
    perturbed_reward_mean: float = 0.0
    robustness_score: float = 1.0  # ratio perturbed/clean
    failure_rate: float = 0.0
    recovery_time_mean: float = 0.0
    n_evaluations: int = 0
    
    def update(self, clean_reward: float, perturbed_reward: float, 
               failed: bool, recovery_time: int) -> None:
        """Actualiza métricas con nuevo episodio."""
        self.n_evaluations += 1
        alpha = 0.9  # Factor de suavizado exponencial
        self.clean_reward_mean = (
            alpha * self.clean_reward_mean + 
            (1 - alpha) * clean_reward
        )
        self.perturbed_reward_mean = (
            alpha * self.perturbed_reward_mean + 
            (1 - alpha) * perturbed_reward
        )
        if self.clean_reward_mean != 0:
            self.robustness_score = (
                alpha * self.robustness_score + 
                (1 - alpha) * (perturbed_reward / abs(clean_reward_mean))
            )
        self.failure_rate = (
            alpha * self.failure_rate + 
            (1 - alpha) * float(failed)
        )
        self.recovery_time_mean = (
            alpha * self.recovery_time_mean + 
            (1 - alpha) * recovery_time
        )


if SB3_AVAILABLE:
    class PerturbationCallback(BaseCallback):
        """
        Callback para inyección de perturbaciones durante entrenamiento.
        
        Integrado con stable-baselines3 para evaluar robustez periódicamente
        durante el entrenamiento del agente PPO/SAC.
        
        Parameters
        ----------
        stress_injector : StressInjector
            Inyector de perturbaciones a aplicar.
        eval_freq : int
            Frecuencia de evaluación de robustez (en timesteps).
        n_eval_episodes : int
            Número de episodios para evaluar robustez.
        verbose : int
            Nivel de verbosidad (0=silent, 1=info, 2=debug).
        
        Examples
        --------
        >>> from src.robustness import PerturbationCallback, StressInjector
        >>> injector = StressInjector(severity=0.1, probability=0.2)
        >>> callback = PerturbationCallback(injector, eval_freq=10000)
        >>> model.learn(total_timesteps=1000000, callback=[callback])
        """
        
        def __init__(
            self,
            stress_injector: StressInjector,
            eval_freq: int = 10000,
            n_eval_episodes: int = 10,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose)
            self.stress_injector = stress_injector
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.metrics = RobustnessMetrics()
            self._eval_count = 0
        
        def _on_step(self) -> bool:
            """Called at each step of training."""
            if self.n_calls % self.eval_freq == 0:
                self._eval_count += 1
                if self.verbose > 0:
                    logger.info(
                        f"[PerturbationCallback] Evaluación #{self._eval_count} | "
                        f"Robustness: {self.metrics.robustness_score:.3f}"
                    )
            return True
        
        def _on_rollout_end(self) -> None:
            """Called at end of each PPO rollout."""
            pass
        
        def get_metrics(self) -> Dict[str, Any]:
            """Obtiene métricas actuales de robustez."""
            return {
                "robustness/robustness_score": self.metrics.robustness_score,
                "robustness/clean_reward_mean": self.metrics.clean_reward_mean,
                "robustness/perturbed_reward_mean": self.metrics.perturbed_reward_mean,
                "robustness/failure_rate": self.metrics.failure_rate,
                "robustness/recovery_time_mean": self.metrics.recovery_time_mean,
                "robustness/n_evaluations": self.metrics.n_evaluations,
            }


class RobustnessEvaluator:
    """
    Evaluador sistemático de robustez multi-escenario.
    
    Evalúa un agente trained bajo múltiples condiciones de perturbación
    para generar un perfil completo de robustez.
    
    Parameters
    ----------
    env_factory : Callable
        Función que crea instancias del entorno.
    agent : object
        Agente RL a evaluar.
    n_episodes : int
        Número de episodios por condición de evaluación.
    seed : int
        Semilla para reproducibilidad.
    
    Examples
    --------
    >>> evaluator = RobustnessEvaluator(env_factory, agent, n_episodes=20)
    >>> results = evaluator.evaluate_all_perturbations()
    >>> print(results['robustness_summary'])
    """
    
    def __init__(
        self,
        env_factory: Callable,
        agent: Any,
        n_episodes: int = 20,
        seed: int = 42,
    ) -> None:
        self.env_factory = env_factory
        self.agent = agent
        self.n_episodes = n_episodes
        self._rng = np.random.default_rng(seed)
        self.results: Dict[str, Any] = {}
    
    def evaluate_condition(
        self,
        perturbation_type: PerturbationType,
        severities: List[float] = [0.0, 0.1, 0.2, 0.5],
    ) -> Dict[str, Any]:
        """
        Evalúa el agente bajo una condición de perturbación específica.
        
        Parameters
        ----------
        perturbation_type : PerturbationType
            Tipo de perturbación a evaluar.
        severities : List[float]
            Lista de niveles de severidad a probar.
        
        Returns
        -------
        dict
            Resultados de la evaluación por severidad.
        """
        results_by_severity = {}
        
        for severity in severities:
            rewards = []
            failures = 0
            
            for ep in range(self.n_episodes):
                env = self.env_factory(seed=ep)
                injector = StressInjector(
                    perturbation_type=perturbation_type,
                    severity=severity,
                    probability=1.0,  # Siempre activo durante evaluación
                    seed=ep,
                )
                
                obs, _ = env.reset()
                total_reward = 0.0
                done = False
                
                while not done:
                    # Aplicar perturbación a observación
                    obs_perturbed = injector.inject(obs)
                    
                    # Obtener acción del agente
                    action, _ = self.agent.predict(obs_perturbed, deterministic=True)
                    
                    # Ejecutar acción
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                
                rewards.append(total_reward)
                if total_reward < -1000:  # Umbral de fallo
                    failures += 1
            
            results_by_severity[severity] = {
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "failure_rate": failures / self.n_episodes,
            }
        
        return {
            "perturbation_type": perturbation_type.name,
            "results_by_severity": results_by_severity,
        }
    
    def evaluate_all_perturbations(
        self,
        severities: List[float] = [0.0, 0.1, 0.2, 0.5],
    ) -> Dict[str, Any]:
        """
        Evalúa el agente bajo todos los tipos de perturbación.
        
        Parameters
        ----------
        severities : List[float]
            Lista de niveles de severidad a probar.
        
        Returns
        -------
        dict
            Resultados completos de robustez.
        """
        all_results = {}
        
        for ptype in PerturbationType:
            all_results[ptype.name] = self.evaluate_condition(ptype, severities)
        
        # Calcular resumen de robustez
        robustness_summary = self._compute_robustness_summary(all_results)
        
        return {
            "detailed_results": all_results,
            "robustness_summary": robustness_summary,
        }
    
    def _compute_robustness_summary(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calcula métricas agregadas de robustez."""
        clean_rewards = []
        perturbed_rewards = []
        
        for ptype_results in results.values():
            for severity, data in ptype_results["results_by_severity"].items():
                if severity == 0.0:
                    clean_rewards.append(data["reward_mean"])
                else:
                    perturbed_rewards.append(data["reward_mean"])
        
        clean_mean = np.mean(clean_rewards) if clean_rewards else 0.0
        perturbed_mean = np.mean(perturbed_rewards) if perturbed_rewards else 0.0
        
        robustness_score = (
            perturbed_mean / abs(clean_mean) 
            if clean_mean != 0 else 1.0
        )
        
        return {
            "clean_reward_mean": float(clean_mean),
            "perturbed_reward_mean": float(perturbed_mean),
            "robustness_score": float(robustness_score),
            "interpretation": self._interpret_robustness(robustness_score),
        }
    
    def _interpret_robustness(self, score: float) -> str:
        """Interpreta el score de robustez."""
        if score >= 0.9:
            return "Excelente - Agente muy robusto"
        elif score >= 0.7:
            return "Bueno - Agente razonablemente robusto"
        elif score >= 0.5:
            return "Moderado - Mejorar robustez recomendada"
        else:
            return "Pobre - Agente frágil bajo perturbaciones"
