"""
Módulo de Función de Recompensa Multiobjetivo.

Implementación exacta de la función de recompensa del Capítulo 4.3.2:
    R_t = -(λ1·Delay_t + λ2·Gini_t + λ3·CVaRα(L_t))

Donde:
    - Delay_t: suma de tiempos de espera por acceso
    - Gini_t: coeficiente de Gini de la distribución de esperas
    - CVaRα: Conditional Value at Risk de las pérdidas

Referencias:
    - Capítulo 4.3.2: Función de recompensa multiobjetivo
    - Apéndice A.4: Valores de hiperparámetros λ1, λ2, λ3
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque


class MultiObjectiveReward:
    """
    Calculadora de recompensa multiobjetivo para control semafórico.
    
    Implementa exactamente la formulación del Capítulo 4.3.2:
        R_t = -(λ1·Delay_t + λ2·Gini_t + λ3·CVaRα(L_t))
    
    Esta clase mantiene el historial necesario para calcular métricas
    deslizantes y proporciona métodos para actualizar y obtener los
    componentes individuales de la recompensa.
    
    Args:
        lambda_delay: Peso para componente de delay (λ1 > 0)
        lambda_gini: Peso para componente de Gini (λ2 > 0)
        lambda_cvar: Peso para componente de CVaR (λ3 > 0)
        alpha: Nivel de confianza para CVaR (por defecto 0.95)
        history_size: Tamaño del buffer deslizante para CVaR
    
    Attributes:
        lambda_delay (float): Peso para delay
        lambda_gini (float): Peso para índice de Gini
        lambda_cvar (float): Peso para CVaR
        alpha (float): Nivel de confianza para CVaR
        loss_history (deque): Historial de pérdidas para cálculo de CVaR
    """
    
    def __init__(
        self,
        lambda_delay: float = 0.4,
        lambda_gini: float = 0.3,
        lambda_cvar: float = 0.3,
        alpha: float = 0.95,
        history_size: int = 100
    ):
        # Validar que los pesos sean positivos (Capítulo 4.3.2)
        if lambda_delay <= 0 or lambda_gini <= 0 or lambda_cvar <= 0:
            raise ValueError(
                "Todos los pesos λ deben ser estrictamente positivos "
                "(Capítulo 4.3.2)"
            )
        
        # Validación explícita: suma de lambdas debe ser 1.0
        lambda_sum = lambda_delay + lambda_gini + lambda_cvar
        assert abs(lambda_sum - 1.0) < 1e-6, (
            f"Los pesos de recompensa deben sumar 1.0. "
            f"Recibido: λ_delay={lambda_delay}, λ_gini={lambda_gini}, λ_cvar={lambda_cvar}. "
            f"Suma: {lambda_sum}"
        )
        
        self.lambda_delay = lambda_delay
        self.lambda_gini = lambda_gini
        self.lambda_cvar = lambda_cvar
        self.alpha = alpha
        self.history_size = history_size
        
        # Buffer deslizante para historial de pérdidas
        self.loss_history = deque(maxlen=history_size)
        
        # Métricas del episodio actual
        self.current_delays = []
        self.current_waits = []
    
    def reset_episode(self):
        """
        Reinicia las métricas al inicio de un nuevo episodio.
        
        Debe llamarse al inicio de cada episodio para limpiar
        las métricas acumuladas.
        """
        self.current_delays = []
        self.current_waits = []
    
    def update_step(
        self,
        delays: List[float],
        waits: List[float]
    ) -> None:
        """
        Actualiza las métricas con datos del step actual.
        
        Args:
            delays: Lista de delays por vehículo/acceso en este step
            waits: Lista de tiempos de espera actuales por acceso
        """
        self.current_delays.extend(delays)
        self.current_waits.extend(waits)
    
    def calculate_delay_component(self) -> float:
        """
        Calcula el componente de delay de la recompensa.
        
        Delay_t = suma de tiempos de espera por acceso
        
        Returns:
            float: Valor del componente de delay (positivo, se negará en R_t)
        """
        if len(self.current_delays) == 0:
            return 0.0
        
        # Delay medio normalizado por número de accesos
        delay_mean = np.mean(self.current_delays)
        return delay_mean
    
    def calculate_gini_component(self) -> float:
        """
        Calcula el coeficiente de Gini de la distribución de esperas.
        
        Fórmula exacta del Capítulo 4.3.2 (optimizada dual):
            G_t = Σ_i Σ_j |w_i,t - w_j,t| / (2n²·w̄_t)
        
        Implementación vectorizada eficiente usando fórmula alternativa:
            G = (2 * Σ(i * x_(i))) / (n * Σx_i) - (n+1)/n
        
        Donde:
            - w_i,t: tiempo de espera del acceso i en tiempo t
            - n: número total de accesos
            - w̄_t: espera media en tiempo t
            - x_(i): valores ordenados ascendentemente
        
        Returns:
            float: Coeficiente de Gini ∈ [0, 1]
        """
        if len(self.current_waits) == 0:
            return 0.0
        
        waits = np.array(self.current_waits)
        n = len(waits)
        
        # Evitar división por cero
        mean_wait = np.mean(waits)
        if mean_wait == 0 or n <= 1:
            return 0.0
        
        # Fórmula dual optimizada (vectorizada)
        sorted_waits = np.sort(waits)
        indices = np.arange(1, n + 1)
        
        gini = (2 * np.sum(indices * sorted_waits)) / (n * np.sum(sorted_waits))
        gini -= (n + 1) / n
        
        # Clamp a [0, 1] por seguridad numérica
        return float(np.clip(gini, 0.0, 1.0))
    
    def calculate_cvar_component(self) -> float:
        """
        Calcula el CVaR (Conditional Value at Risk) de las pérdidas.
        
        Fórmula del Capítulo 4.3.2:
            CVaRα(L) = E[L | L ≥ VaRα(L)]
        
        Implementación con ventana deslizante (collections.deque maxlen=100):
            - Mantiene historial limitado para cálculo eficiente
            - Actualiza incrementalmente en cada paso
        
        Donde:
            - VaRα(L): percentil α de la distribución de pérdidas
            - L: pérdidas (delays en nuestro caso)
            - α: nivel de confianza (default 0.95)
        
        Returns:
            float: Valor de CVaR (≥ VaRα)
        """
        if len(self.loss_history) == 0:
            return 0.0
        
        losses = np.array(list(self.loss_history))
        
        # Calcular VaR como percentil α
        var_alpha = np.percentile(losses, self.alpha * 100)
        
        # CVaR es la esperanza condicional sobre el umbral VaR
        tail_losses = losses[losses >= var_alpha]
        
        if len(tail_losses) == 0:
            return var_alpha
        
        cvar = np.mean(tail_losses)
        return float(cvar)
    
    def calculate_reward(
        self,
        delays: Optional[List[float]] = None,
        waits: Optional[List[float]] = None
    ) -> Tuple[float, dict]:
        """
        Calcula la recompensa multiobjetivo completa.
        
        Fórmula exacta del Capítulo 4.3.2:
            R_t = -(λ1·Delay_t + λ2·Gini_t + λ3·CVaRα(L_t))
        
        Args:
            delays: Delays actuales (opcional, usa los acumulados si None)
            waits: Wait times actuales (opcional, usa los acumulados si None)
        
        Returns:
            tuple: (recompensa_total, diccionario_de_componentes)
                - recompensa_total: valor escalar de R_t
                - diccionario_de_componentes: {'delay', 'gini', 'cvar', 'total_loss'}
        """
        # Si se proporcionan nuevos datos, actualizar
        if delays is not None:
            self.current_delays = delays
        if waits is not None:
            self.current_waits = waits
        
        # Calcular componentes individuales
        delay_component = self.calculate_delay_component()
        gini_component = self.calculate_gini_component()
        cvar_component = self.calculate_cvar_component()
        
        # Pérdida total (suma ponderada de componentes)
        total_loss = (
            self.lambda_delay * delay_component +
            self.lambda_gini * gini_component +
            self.lambda_cvar * cvar_component
        )
        
        # Actualizar historial de pérdidas para CVaR
        self.loss_history.append(total_loss)
        
        # Recompensa = -pérdida (minimizar pérdida = maximizar recompensa)
        reward = -total_loss
        
        # Diccionario de componentes para logging
        components = {
            'delay': delay_component,
            'gini': gini_component,
            'cvar': cvar_component,
            'total_loss': total_loss,
            'reward': reward
        }
        
        return reward, components
    
    def get_metrics_summary(self) -> dict:
        """
        Obtiene un resumen de todas las métricas actuales.
        
        Returns:
            dict: Diccionario con todas las métricas calculadas
        """
        delay = self.calculate_delay_component()
        gini = self.calculate_gini_component()
        cvar = self.calculate_cvar_component()
        
        return {
            'delay_mean': delay,
            'gini': gini,
            'cvar_alpha': cvar,
            'lambda_delay': self.lambda_delay,
            'lambda_gini': self.lambda_gini,
            'lambda_cvar': self.lambda_cvar,
            'alpha': self.alpha,
            'history_size': len(self.loss_history)
        }


def gini_coefficient(values: np.ndarray) -> float:
    """
    Calcula el coeficiente de Gini para un array de valores.
    
    Implementación vectorizada eficiente usando la fórmula:
        G = (2 * Σ(i * x_(i))) / (n * Σx_i) - (n+1)/n
    
    Donde x_(i) son los valores ordenados ascendentemente.
    
    Args:
        values: Array de valores no negativos
    
    Returns:
        float: Coeficiente de Gini ∈ [0, 1]
    """
    if len(values) == 0:
        return 0.0
    
    values = np.asarray(values)
    values = values[values >= 0]  # Filtrar valores negativos
    
    if len(values) <= 1 or np.sum(values) == 0:
        return 0.0
    
    # Ordenar valores ascendentemente
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Índices 1-based para la fórmula
    indices = np.arange(1, n + 1)
    
    # Fórmula de Gini
    gini = (2 * np.sum(indices * sorted_values)) / (n * np.sum(sorted_values))
    gini -= (n + 1) / n
    
    return np.clip(gini, 0.0, 1.0)


def cvar_calculation(
    losses: np.ndarray,
    alpha: float = 0.95
) -> Tuple[float, float]:
    """
    Calcula VaR y CVaR para un array de pérdidas.
    
    Fórmulas del Capítulo 4.3.2:
        VaRα(L) = percentil_α(L)
        CVaRα(L) = E[L | L ≥ VaRα(L)]
    
    Args:
        losses: Array de pérdidas (valores positivos)
        alpha: Nivel de confianza (por defecto 0.95)
    
    Returns:
        tuple: (var_alpha, cvar_alpha)
    """
    if len(losses) == 0:
        return 0.0, 0.0
    
    losses = np.asarray(losses)
    
    # Calcular VaR como percentil
    var_alpha = np.percentile(losses, alpha * 100)
    
    # Calcular CVaR como esperanza condicional
    tail_losses = losses[losses >= var_alpha]
    
    if len(tail_losses) == 0:
        return var_alpha, var_alpha
    
    cvar_alpha = np.mean(tail_losses)
    
    return var_alpha, cvar_alpha
