"""
Agente PPO (Proximal Policy Optimization) para control semafórico.

Implementación de agente RL basado en PPO con soporte para métricas de riesgo
y equidad distributiva, siguiendo la formulación del Capítulo 4.3.2 y
Apéndice A.4 de la tesis doctoral.

Referencias:
    - Capítulo 4.3.2: Función de recompensa multiobjetivo
    - Apéndice A.4: Hiperparámetros de entrenamiento PPO
    - Schulman et al. (2017): Proximal Policy Optimization Algorithms
"""

import torch
import numpy as np
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from rl_agent.callbacks import RiskMetricsCallback


class PPOAgent:
    """
    Agente PPO especializado para control semafórico inteligente.
    
    Esta clase envuelve la implementación de PPO de stable-baselines3
    con configuraciones específicas para el problema de control semafórico,
    incluyendo:
    - Forzado de ejecución en CPU (sin GPU)
    - Callbacks personalizados para métricas de riesgo
    - Configuración de hiperparámetros según Apéndice A.4
    
    Args:
        env: Entorno de entrenamiento (VecEnv o gym.Env)
        learning_rate: Tasa de aprendizaje (por defecto 3e-4)
        n_steps: Número de steps por rollout (por defecto 2048)
        batch_size: Tamaño de batch para actualización (por defecto 64)
        n_epochs: Número de épocas de actualización (por defecto 10)
        gamma: Factor de descuento (por defecto 0.99)
        gae_lambda: Parámetro lambda para GAE (por defecto 0.95)
        clip_range: Límite de clipping para PPO (por defecto 0.2)
        ent_coef: Coeficiente de entropía (por defecto 0.01)
        verbose: Nivel de verbosidad (0, 1, 2)
        tensorboard_log: Ruta para logs de TensorBoard
        device: Dispositivo de computación (forzado a 'cpu')
    """
    
    def __init__(
        self,
        env: Union[VecEnv, Any],
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 0,
        tensorboard_log: Optional[str] = None,
        device: str = "cpu"
    ):
        # Forzar dispositivo CPU explícitamente
        self.device = torch.device("cpu")
        
        # Validar que no se use GPU
        if device != "cpu":
            print(f"Advertencia: dispositivo '{device}' ignorado. Forzando CPU.")
        
        # Configurar política personalizada si es necesario
        policy_kwargs = {
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
        }
        
        # Crear modelo PPO
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=self.device,
            policy_kwargs=policy_kwargs
        )
        
        # Registrar configuración
        self.config = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef
        }
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False
    ) -> "PPOAgent":
        """
        Entrena el agente PPO.
        
        Args:
            total_timesteps: Número total de steps de entrenamiento
            callback: Callback personalizado (ej. RiskMetricsCallback)
            log_interval: Frecuencia de logging
            tb_log_name: Nombre del log en TensorBoard
            reset_num_timesteps: Resetear contador de timesteps
            progress_bar: Mostrar barra de progreso
            
        Returns:
            self: Referencia al propio agente para encadenamiento
        """
        # Añadir callback de métricas de riesgo si no se proporciona
        if callback is None:
            callback = RiskMetricsCallback(verbose=1)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
        
        return self
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta del archivo de guardado (sin extensión)
        """
        self.model.save(path)
        print(f"Modelo guardado en {path}.zip")
    
    @classmethod
    def load(cls, path: str, env: Optional[Any] = None) -> "PPOAgent":
        """
        Carga un modelo pre-entrenado.
        
        Args:
            path: Ruta del archivo del modelo (con o sin .zip)
            env: Entorno opcional para cargar con el modelo
            
        Returns:
            PPOAgent: Instancia cargada del agente
        """
        # Añadir extensión .zip si no está presente
        if not path.endswith('.zip'):
            path = path + '.zip'
        
        model = PPO.load(path, env=env, device="cpu")
        
        # Crear instancia wrapper
        agent = cls.__new__(cls)
        agent.model = model
        agent.device = torch.device("cpu")
        agent.config = {}
        
        return agent
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predice la acción para una observación dada.
        
        Args:
            observation: Observación del entorno (array o tensor)
            deterministic: Usar política determinística (True para evaluación)
            
        Returns:
            tuple: (acción, valores de valor/acción si disponibles)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, _
    
    def evaluate(
        self,
        eval_env: Any,
        n_eval_episodes: int = 5,
        deterministic: bool = True
    ) -> tuple:
        """
        Evalúa el agente en un entorno de evaluación.
        
        Args:
            eval_env: Entorno de evaluación
            n_eval_episodes: Número de episodios para evaluación
            deterministic: Usar política determinística
            
        Returns:
            tuple: (reward_mean, reward_std)
        """
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic
        )
        return mean_reward, std_reward
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del agente.
        
        Returns:
            Dict: Diccionario con los hiperparámetros del agente
        """
        return self.config.copy()
