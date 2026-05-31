"""
Métricas de robustez para evaluación de agentes RL bajo perturbaciones.
Basado en patrones de CAREL para evaluación formal bajo incertidumbre.

@ref: fmpr/CAREL (https://github.com/fmpr/CAREL)
"""
import numpy as np
from typing import List, Optional


class RobustnessMetrics:
    """
    Métricas cuantitativas de robustez para sistemas de control de tráfico.
    
    Incluye:
    - Performance Drop: Caída porcentual bajo estrés
    - Recovery Time: Pasos para recuperar performance baseline
    - Stability Index: Inversa de la varianza de rewards
    """
    
    @staticmethod
    def performance_drop(baseline_performance: float, stressed_performance: float) -> float:
        """
        Calcula la caída porcentual de performance bajo perturbaciones.
        
        Args:
            baseline_performance: Performance sin perturbaciones (ej: reward promedio).
            stressed_performance: Performance con perturbaciones aplicadas.
            
        Returns:
            Porcentaje de caída (0-100). Positivo indica degradación.
        """
        if baseline_performance == 0:
            return 0.0
            
        drop = ((baseline_performance - stressed_performance) / abs(baseline_performance)) * 100
        return max(0.0, drop)  # No reportar mejoras como "drop negativo"
    
    @staticmethod
    def recovery_time(
        rewards: List[float], 
        baseline: float, 
        threshold: float = 0.95
    ) -> Optional[int]:
        """
        Calcula el número de episodios/pasos para recuperar performance baseline.
        
        Args:
            rewards: Lista de rewards por episodio después de la perturbación.
            baseline: Nivel de performance objetivo.
            threshold: Umbral relativo al baseline (ej: 0.95 = 95% de recuperación).
            
        Returns:
            Índice del primer episodio que alcanza el umbral, o None si no se recupera.
        """
        target = baseline * threshold
        
        for i, reward in enumerate(rewards):
            if reward >= target:
                return i
                
        return None  # No se recuperó en el horizonte observado
    
    @staticmethod
    def stability_index(rewards: List[float]) -> float:
        """
        Calcula un índice de estabilidad basado en la varianza inversa de rewards.
        
        Un índice alto indica baja varianza (alta estabilidad).
        Normalizado a [0, 1] donde 1 es perfectamente estable.
        
        Args:
            rewards: Lista de rewards por episodio.
            
        Returns:
            Índice de estabilidad en [0, 1].
        """
        if len(rewards) < 2:
            return 1.0  # Sin varianza con un solo dato
            
        variance = np.var(rewards)
        mean_reward = np.mean(rewards)
        
        if mean_reward == 0:
            return 0.0
            
        # Coeficiente de variación invertido y normalizado
        cv = np.sqrt(variance) / abs(mean_reward)
        
        # Transformar a [0, 1]: CV=0 → 1.0, CV→∞ → 0.0
        stability = 1.0 / (1.0 + cv)
        
        return stability
    
    @staticmethod
    def worst_case_performance(rewards: List[float], percentile: float = 10.0) -> float:
        """
        Calcula el performance en el peor caso (percentil inferior).
        
        Args:
            rewards: Lista de rewards por episodio.
            percentile: Percentil a calcular (ej: 10 = peor 10%).
            
        Returns:
            Valor del reward en el percentil especificado.
        """
        return float(np.percentile(rewards, percentile))
    
    @staticmethod
    def failure_rate(rewards: List[float], failure_threshold: float = 0.0) -> float:
        """
        Calcula la tasa de fallos (episodios con reward below threshold).
        
        Args:
            rewards: Lista de rewards por episodio.
            failure_threshold: Umbral considerado como "fallo".
            
        Returns:
            Fracción de episodios que fallaron (0-1).
        """
        failures = sum(1 for r in rewards if r < failure_threshold)
        return failures / len(rewards) if len(rewards) > 0 else 0.0
    
    @staticmethod
    def robustness_score(
        baseline: float,
        stressed_rewards: List[float],
        weights: dict = None
    ) -> float:
        """
        Calcula un score compuesto de robustez (0-100).
        
        Combina múltiples métricas en un score único:
        - Performance retention (40%)
        - Stability (30%)
        - Worst-case performance (20%)
        - Failure rate inverso (10%)
        
        Args:
            baseline: Performance baseline sin estrés.
            stressed_rewards: Rewards bajo condiciones de estrés.
            weights: Pesos personalizados para cada componente.
            
        Returns:
            Score de robustez en [0, 100].
        """
        if weights is None:
            weights = {
                'performance': 0.4,
                'stability': 0.3,
                'worst_case': 0.2,
                'failure_rate': 0.1
            }
        
        # 1. Performance retention
        avg_stressed = np.mean(stressed_rewards)
        perf_retention = max(0, (avg_stressed / baseline) * 100) if baseline != 0 else 0
        
        # 2. Stability
        stability = RobustnessMetrics.stability_index(stressed_rewards) * 100
        
        # 3. Worst-case (percentil 10)
        worst = RobustnessMetrics.worst_case_performance(stressed_rewards, 10.0)
        worst_score = max(0, (worst / baseline) * 100) if baseline != 0 else 0
        
        # 4. Failure rate inverso
        fail_rate = RobustnessMetrics.failure_rate(stressed_rewards, baseline * 0.5)
        fail_score = (1.0 - fail_rate) * 100
        
        # Score ponderado
        score = (
            weights['performance'] * perf_retention +
            weights['stability'] * stability +
            weights['worst_case'] * worst_score +
            weights['failure_rate'] * fail_score
        )
        
        return min(100.0, max(0.0, score))
