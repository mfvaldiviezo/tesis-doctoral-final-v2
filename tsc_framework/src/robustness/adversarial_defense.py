"""
AdversarialDefense - Defensa Adversarial y Regularización Lipschitz.
======================================================================

@ref: [ERNIE Paper] + [https://github.com/abukharin3/ERNIE]

Este módulo implementa técnicas de defensa adversarial inspiradas en ERNIE
para mejorar la robustez del agente RL frente a perturbaciones maliciosas
o naturales en las observaciones.

Componentes:
    - AdversarialDefense: Generación de ejemplos adversariales (FGSM-style)
    - LipschitzRegularizer: Penalización de gradiente para suavidad

Autor: Doctoral Researcher
Versión: 1.0.0 (Integración SOTA SLR-2026)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore

logger = logging.getLogger(__name__)


class AdversarialDefense:
    """
    Generador de ejemplos adversariales para entrenamiento defensivo.
    
    Implementa Fast Gradient Sign Method (FGSM) adaptado para políticas RL,
    siguiendo el enfoque de ERNIE para mejorar robustez.
    
    Parameters
    ----------
    epsilon : float
        Magnitud máxima de la perturbación adversarial [0, 1].
    alpha : float
        Paso de actualización para ataques iterativos.
    n_steps : int
        Número de pasos para ataques iterativos (PGD-style).
    
    Examples
    --------
    >>> defense = AdversarialDefense(epsilon=0.05)
    >>> obs_adv = defense.generate_adversarial_obs(obs, model)
    """
    
    def __init__(
        self,
        epsilon: float = 0.05,
        alpha: float = 0.01,
        n_steps: int = 10,
    ) -> None:
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch no disponible. AdversarialDefense desactivado.")
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_steps = n_steps
        self._enabled = TORCH_AVAILABLE
    
    def generate_adversarial_obs(
        self,
        observation: np.ndarray,
        model: Any,
        action_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Genera una observación adversarial que maximiza el error del modelo.
        
        Parameters
        ----------
        observation : np.ndarray
            Observación original limpia.
        model : nn.Module
            Modelo de política o valor a atacar.
        action_target : np.ndarray, optional
            Acción objetivo (si None, usa la predicción del modelo).
        
        Returns
        -------
        np.ndarray
            Observación adversarial perturbada.
        """
        if not self._enabled or not TORCH_AVAILABLE:
            return observation.copy()
        
        with torch.enable_grad():
            # Convertir a tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            obs_tensor.requires_grad_(True)
            
            # Forward pass
            if hasattr(model, 'policy'):
                output = model.policy(obs_tensor)
            else:
                output = model(obs_tensor)
            
            # Calcular pérdida (negar log-probabilidad de acción)
            if action_target is None:
                # Ataque no dirigido: maximizar entropía
                if isinstance(output, tuple):
                    dist = output[0]
                else:
                    dist = output
                loss = -dist.entropy().mean()
            else:
                # Ataque dirigido: minimizar probabilidad de acción target
                action_tensor = torch.LongTensor([action_target])
                if isinstance(output, tuple):
                    dist = output[0]
                else:
                    dist = output
                log_prob = dist.log_prob(action_tensor)
                loss = -log_prob.mean()
            
            # Backward pass
            loss.backward()
            
            # FGSM: perturbar en dirección del gradiente
            grad_sign = obs_tensor.grad.sign()
            adv_obs = obs_tensor + self.epsilon * grad_sign
            
            # Clippear a rango válido [0, 1]
            adv_obs = torch.clamp(adv_obs, 0.0, 1.0)
            
            return adv_obs.squeeze(0).detach().numpy()
    
    def pgd_attack(
        self,
        observation: np.ndarray,
        model: Any,
    ) -> np.ndarray:
        """
        Projected Gradient Descent attack (ataque iterativo más fuerte).
        
        Parameters
        ----------
        observation : np.ndarray
            Observación original.
        model : nn.Module
            Modelo a atacar.
        
        Returns
        -------
        np.ndarray
            Observación adversarial después de n_steps iteraciones.
        """
        if not self._enabled or not TORCH_AVAILABLE:
            return observation.copy()
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        adv_obs = obs_tensor.clone()
        
        for _ in range(self.n_steps):
            adv_obs.requires_grad_(True)
            
            if hasattr(model, 'policy'):
                output = model.policy(adv_obs)
            else:
                output = model(adv_obs)
            
            # Pérdida: negativa log-probabilidad
            if isinstance(output, tuple):
                dist = output[0]
            else:
                dist = output
            
            # Usar entropía como proxy de incertidumbre
            loss = -dist.entropy().mean()
            
            grad = torch.autograd.grad(loss, adv_obs)[0]
            adv_obs = adv_obs + self.alpha * grad.sign()
            
            # Proyectar a bola épsilon
            diff = adv_obs - obs_tensor
            diff = torch.clamp(diff, -self.epsilon, self.epsilon)
            adv_obs = torch.clamp(obs_tensor + diff, 0.0, 1.0)
        
        return adv_obs.squeeze(0).detach().numpy()
    
    def adversarial_training_step(
        self,
        observation: np.ndarray,
        model: Any,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Realiza un paso de entrenamiento adversarial defensivo.
        
        Parameters
        ----------
        observation : np.ndarray
            Observación original.
        model : nn.Module
            Modelo a entrenar.
        optimizer : torch.optim.Optimizer
            Optimizador para actualizar parámetros.
        
        Returns
        -------
        float
            Pérdida del entrenamiento adversarial.
        """
        if not self._enabled or not TORCH_AVAILABLE:
            return 0.0
        
        # Generar ejemplo adversarial
        adv_obs = self.generate_adversarial_obs(observation, model)
        
        # Entrenar con ejemplo adversarial
        optimizer.zero_grad()
        
        if hasattr(model, 'policy'):
            output = model.policy(torch.FloatTensor(adv_obs).unsqueeze(0))
        else:
            output = model(torch.FloatTensor(adv_obs).unsqueeze(0))
        
        # Loss de regularización (maximizar entropía para robustez)
        if isinstance(output, tuple):
            dist = output[0]
        else:
            dist = output
        
        loss = -dist.entropy().mean()
        loss.backward()
        optimizer.step()
        
        return float(loss.item())


class LipschitzRegularizer:
    """
    Regularizador Lipschitz para suavizar la política.
    
    Penaliza grandes cambios en la salida del modelo ante pequeñas
    perturbaciones en la entrada, mejorando la estabilidad y robustez.
    
    @ref: [ERNIE Paper] + [https://github.com/abukharin3/ERNIE]
    
    Parameters
    ----------
    lambda_lip : float
        Peso de la penalización Lipschitz en la función de pérdida.
    n_samples : int
        Número de muestras aleatorias para estimar constante Lipschitz.
    
    Examples
    --------
    >>> regularizer = LipschitzRegularizer(lambda_lip=0.01)
    >>> lip_loss = regularizer.compute_penalty(model, obs_batch)
    >>> total_loss = policy_loss + lip_loss
    """
    
    def __init__(
        self,
        lambda_lip: float = 0.01,
        n_samples: int = 5,
    ) -> None:
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch no disponible. LipschitzRegularizer desactivado.")
        
        self.lambda_lip = lambda_lip
        self.n_samples = n_samples
        self._enabled = TORCH_AVAILABLE
    
    def compute_penalty(
        self,
        model: Any,
        observations: np.ndarray,
    ) -> float:
        """
        Calcula la penalización Lipschitz para un batch de observaciones.
        
        Parameters
        ----------
        model : nn.Module
            Modelo cuya suavidad se quiere regularizar.
        observations : np.ndarray
            Batch de observaciones (batch_size, obs_dim).
        
        Returns
        -------
        float
            Valor de la penalización Lipschitz.
        """
        if not self._enabled or not TORCH_AVAILABLE:
            return 0.0
        
        obs_tensor = torch.FloatTensor(observations)
        batch_size = obs_tensor.shape[0]
        
        total_penalty = 0.0
        
        for _ in range(self.n_samples):
            # Generar perturbación aleatoria pequeña
            perturbation = torch.randn_like(obs_tensor) * 0.01
            perturbed_obs = obs_tensor + perturbation
            
            # Evaluar modelo en original y perturbado
            if hasattr(model, 'policy'):
                output_orig = model.policy(obs_tensor)
                output_pert = model.policy(perturbed_obs)
            else:
                output_orig = model(obs_tensor)
                output_pert = model(perturbed_obs)
            
            # Extraer medias de distribución
            if isinstance(output_orig, tuple):
                mean_orig = output_orig[0].mean
                mean_pert = output_pert[0].mean
            else:
                mean_orig = output_orig
                mean_pert = output_pert
            
            # Calcular ratio cambio_output / cambio_input
            output_diff = torch.norm(mean_orig - mean_pert, p=2, dim=-1)
            input_diff = torch.norm(perturbation, p=2, dim=-1)
            
            # Evitar división por cero
            eps = 1e-8
            lip_constant = output_diff / (input_diff + eps)
            
            # Penalizar constantes Lipschitz grandes
            penalty = torch.relu(lip_constant - 1.0).mean()
            total_penalty += penalty
        
        avg_penalty = total_penalty / self.n_samples
        
        return float(avg_penalty.item()) * self.lambda_lip
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del regularizador."""
        return {
            "lambda_lipschitz": self.lambda_lip,
            "n_samples": self.n_samples,
            "enabled": self._enabled,
        }
