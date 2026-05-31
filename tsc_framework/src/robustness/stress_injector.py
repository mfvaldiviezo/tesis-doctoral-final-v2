"""
StressInjector - Inyección de Perturbaciones para Evaluación de Robustez.
==========================================================================

@ref: [CAREL Paper] + [https://github.com/fmpr/CAREL]
@ref: [ERNIE Paper] + [https://github.com/abukharin3/ERNIE]

Este módulo implementa la inyección sistemática de perturbaciones en el entorno
de simulación SUMO para evaluar la robustez del agente RL bajo condiciones de
incertidumbre, siguiendo las mejores prácticas de CAREL y ERNIE (SLR 2026).

Tipos de Perturbación Implementados:
    1. SENSOR_NOISE: Ruido gaussiano en observaciones (simula errores de detección)
    2. DEMAND_SURGE: Cambios abruptos en la demanda de tráfico
    3. PHASE_FAILURE: Fallos en la ejecución de fases semafóricas
    4. COMMUNICATION_DELAY: Retrasos en la aplicación de acciones

Uso en TSCEnv:
    >>> from src.robustness import StressInjector, PerturbationType
    >>> injector = StressInjector(
    ...     perturbation_type=PerturbationType.SENSOR_NOISE,
    ...     severity=0.1,
    ...     probability=0.2
    ... )
    >>> perturbed_obs = injector.inject(observation)

Autor: Doctoral Researcher
Versión: 1.0.0 (Integración SOTA SLR-2026)
"""

from __future__ import annotations

import numpy as np
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


class PerturbationType(Enum):
    """Tipos de perturbación soportados."""
    SENSOR_NOISE = auto()        # Ruido en observaciones
    DEMAND_SURGE = auto()        # Picos de demanda
    PHASE_FAILURE = auto()       # Fallo en ejecución de fase
    COMMUNICATION_DELAY = auto() # Retraso en acción
    DETECTION_FAILURE = auto()   # Pérdida parcial de detectores


@dataclass
class PerturbationConfig:
    """Configuración de perturbación."""
    perturbation_type: PerturbationType
    severity: float = 0.1         # Magnitud de la perturbación [0, 1]
    probability: float = 0.2      # Probabilidad de ocurrencia [0, 1]
    duration_steps: int = 10      # Duración en pasos de simulación
    start_step: int = 0           # Paso de inicio relativo al episodio
    
    def __post_init__(self):
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"Severity must be in [0, 1], got {self.severity}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")


class StressInjector:
    """
    Inyector de estrés para evaluación de robustez en control semafórico.
    
    Implementa patrones de perturbación basados en CAREL y ERNIE para evaluar
    la resiliencia del agente RL bajo condiciones adversas.
    
    Parameters
    ----------
    perturbation_type : PerturbationType
        Tipo de perturbación a aplicar.
    severity : float
        Magnitud de la perturbación (0.0 = sin efecto, 1.0 = máximo impacto).
    probability : float
        Probabilidad de que la perturbación ocurra en cada paso.
    duration_steps : int
        Número de pasos que persiste la perturbación una vez activada.
    seed : int, optional
        Semilla para reproducibilidad.
    
    Examples
    --------
    >>> injector = StressInjector(
    ...     perturbation_type=PerturbationType.SENSOR_NOISE,
    ...     severity=0.15,
    ...     probability=0.3,
    ...     seed=42
    ... )
    >>> obs_perturbed = injector.inject(obs_clean)
    """
    
    def __init__(
        self,
        perturbation_type: PerturbationType = PerturbationType.SENSOR_NOISE,
        severity: float = 0.1,
        probability: float = 0.2,
        duration_steps: int = 10,
        seed: int = 42,
    ) -> None:
        self.config = PerturbationConfig(
            perturbation_type=perturbation_type,
            severity=severity,
            probability=probability,
            duration_steps=duration_steps,
        )
        self._rng = np.random.default_rng(seed)
        self._active = False
        self._steps_remaining = 0
        self._total_injections = 0
        self._episode_step = 0
    
    def reset(self, episode_step: int = 0) -> None:
        """
        Reinicia el estado del inyector al inicio de un episodio.
        
        Parameters
        ----------
        episode_step : int
            Paso inicial del episodio (para sincronización).
        """
        self._active = False
        self._steps_remaining = 0
        self._episode_step = episode_step
    
    def should_activate(self) -> bool:
        """Determina si la perturbación debe activarse en este paso."""
        if self._active:
            return False
        return self._rng.random() < self.config.probability
    
    def inject(self, observation: np.ndarray, action: Optional[int] = None) -> np.ndarray:
        """
        Aplica la perturbación a la observación o acción.
        
        Parameters
        ----------
        observation : np.ndarray
            Observación original del entorno (shape=(34,) para TSCEnv).
        action : int, optional
            Acción original (usada para perturbaciones de tipo PHASE_FAILURE).
        
        Returns
        -------
        np.ndarray
            Observación perturbada.
        """
        self._episode_step += 1
        
        # Actualizar estado de actividad
        if self._active:
            self._steps_remaining -= 1
            if self._steps_remaining <= 0:
                self._active = False
        elif self.should_activate():
            self._active = True
            self._steps_remaining = self.config.duration_steps
            self._total_injections += 1
        
        # Aplicar perturbación según tipo
        if not self._active:
            return observation.copy()
        
        if self.config.perturbation_type == PerturbationType.SENSOR_NOISE:
            return self._inject_sensor_noise(observation)
        elif self.config.perturbation_type == PerturbationType.DETECTION_FAILURE:
            return self._inject_detection_failure(observation)
        else:
            # Para otros tipos que no modifican observación directamente
            return observation.copy()
    
    def _inject_sensor_noise(self, observation: np.ndarray) -> np.ndarray:
        """
        Inyecta ruido gaussiano aditivo en la observación.
        
        Simula errores de medición en los detectores de tráfico.
        
        Parameters
        ----------
        observation : np.ndarray
            Observación original normalizada [0, 1].
        
        Returns
        -------
        np.ndarray
            Observación con ruido, clippeada a [0, 1].
        """
        noise_std = self.config.severity * 0.5  # Escalar severidad a std dev
        noise = self._rng.normal(0, noise_std, size=observation.shape)
        perturbed = observation + noise
        # Mantener dentro de bounds válidos
        return np.clip(perturbed, 0.0, 1.0).astype(np.float32)
    
    def _inject_detection_failure(self, observation: np.ndarray) -> np.ndarray:
        """
        Simula fallo parcial de detectores (cero lecturas aleatorias).
        
        Parameters
        ----------
        observation : np.ndarray
            Observación original.
        
        Returns
        -------
        np.ndarray
            Observación con algunos componentes puestos a cero.
        """
        perturbed = observation.copy()
        n_features = len(observation)
        # Número de features a fallar proporcional a severity
        n_failures = max(1, int(n_features * self.config.severity))
        failure_indices = self._rng.choice(n_features, size=n_failures, replace=False)
        perturbed[failure_indices] = 0.0
        return perturbed
    
    def perturb_action(self, action: int, n_phases: int = 4) -> int:
        """
        Perturba una acción (fase semafórica) antes de su ejecución.
        
        Usado para perturbaciones de tipo PHASE_FAILURE y COMMUNICATION_DELAY.
        
        Parameters
        ----------
        action : int
            Acción original (índice de fase).
        n_phases : int
            Número total de fases disponibles.
        
        Returns
        -------
        int
            Acción perturbada.
        """
        if not self._active:
            return action
        
        if self.config.perturbation_type == PerturbationType.PHASE_FAILURE:
            # Ejecutar fase aleatoria en lugar de la solicitada
            return int(self._rng.integers(0, n_phases))
        elif self.config.perturbation_type == PerturbationType.COMMUNICATION_DELAY:
            # Retornar acción anterior (simular delay) - se maneja externamente
            return action
        
        return action
    
    def get_demand_multiplier(self) -> float:
        """
        Obtiene el multiplicador de demanda para perturbaciones DEMAND_SURGE.
        
        Returns
        -------
        float
            Multiplicador de demanda (>1 para surge, <1 para reducción).
        """
        if not self._active:
            return 1.0
        
        if self.config.perturbation_type == PerturbationType.DEMAND_SURGE:
            # Aumento de demanda proporcional a severity
            return 1.0 + self.config.severity * 2.0  # Hasta 3x demanda
        
        return 1.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de uso del inyector.
        
        Returns
        -------
        dict
            Diccionario con métricas de perturbación.
        """
        return {
            "perturbation_type": self.config.perturbation_type.name,
            "severity": self.config.severity,
            "probability": self.config.probability,
            "total_injections": self._total_injections,
            "currently_active": self._active,
            "steps_remaining": max(0, self._steps_remaining),
        }
