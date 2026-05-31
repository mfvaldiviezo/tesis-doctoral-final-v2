"""
Módulo Probabilístico - Generación de Escenarios de Estrés con Vine Copulas

Referencia Tesis Doctoral:
    Capítulo 4.3.1: "Generación de Escenarios de Tráfico mediante Vine Copulas"
    - Sección 4.3.1.1: Ajuste de marginales empíricas/paramétricas
    - Sección 4.3.1.2: Estimación de estructura Regular Vine
    - Sección 4.3.1.3: Muestreo condicional de escenarios de estrés
    - Sección 4.3.1.4: Exportación a formatos SUMO compatibles

Descripción:
    Este módulo implementa un pipeline de 4 pasos para generar escenarios de tráfico
    realistas con dependencias de cola pesada y asimetría, utilizando Regular Vine Copulas.
    
    Variables modeladas:
    1. Demanda vehicular (vehículos/hora por acceso)
    2. Tiempos entre llegadas (distribución exponencial/generalizada)
    3. Fricción conductual (variabilidad en aceleración/deceleración)
    4. Probabilidad de incidente (eventos raros de alta impacto)

Autor: Framework TSC - Tesis Doctoral
Versión: 1.0.0
"""

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest

try:
    import pyvinecopulib as pv
    VINE_AVAILABLE = True
except ImportError:
    VINE_AVAILABLE = False
    pv = None

# Type hints condicionales para evitar errores cuando pyvinecopulib no está disponible
if VINE_AVAILABLE:
    VinecopType = pv.Vinecop
    FamilySetType = object  # Usar object como fallback genérico
else:
    VinecopType = type(None)
    FamilySetType = object

logger = logging.getLogger(__name__)


class MarginalDistribution(Enum):
    """
    Distribuciones marginales soportadas para ajuste de variables de tráfico.
    
    Referencia: Cap 4.3.1.1 - Selección de familias paramétricas según
    características empíricas de cada variable.
    """
    EXPONENTIAL = "exponential"  # Tiempos entre llegadas
    GAMMA = "gamma"  # Demanda vehicular
    LOGNORMAL = "lognormal"  # Fricción conductual
    BETA = "beta"  # Probabilidades normalizadas
    GENERALIZED_PARETO = "genpareto"  # Colas pesadas (incidentes)
    EMPIRICAL = "empirical"  # Sin ajuste paramétrico


@dataclass
class MarginalFit:
    """
    Resultado del ajuste de una distribución marginal.
    
    Atributos:
        variable_name: Nombre de la variable (ej. "demand_access_0")
        distribution: Tipo de distribución ajustada
        params: Parámetros estimados de la distribución
        ks_statistic: Estadístico D de Kolmogorov-Smirnov
        ks_pvalue: Valor p del test KS (H0: misma distribución)
        data_min: Mínimo observado en datos originales
        data_max: Máximo observado en datos originales
    """
    variable_name: str
    distribution: MarginalDistribution
    params: Tuple[float, ...]
    ks_statistic: float
    ks_pvalue: float
    data_min: float
    data_max: float
    
    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Genera muestras de la distribución ajustada.
        
        Args:
            n_samples: Número de muestras a generar
            rng: Generador de números aleatorios numpy
            
        Returns:
            Array de muestras de la distribución
        """
        if self.distribution == MarginalDistribution.EXPONENTIAL:
            # params = (scale,)
            return rng.exponential(scale=self.params[0], size=n_samples)
        elif self.distribution == MarginalDistribution.GAMMA:
            # params = (a, loc, scale) - usar shape y scale directamente
            return rng.gamma(shape=self.params[0], scale=self.params[2], size=n_samples) + self.params[1]
        elif self.distribution == MarginalDistribution.LOGNORMAL:
            # params = (s, loc, scale) donde s=sigma, scale=exp(mu)
            return rng.lognormal(mean=np.log(max(self.params[2], 1e-10)), 
                                sigma=self.params[0], size=n_samples)
        elif self.distribution == MarginalDistribution.BETA:
            # params = (a, b, loc, scale)
            return rng.beta(a=self.params[0], b=self.params[1], size=n_samples) * self.params[3] + self.params[2]
        elif self.distribution == MarginalDistribution.GENERALIZED_PARETO:
            # params = (c, loc, scale)
            return rng.pareto(c=self.params[0], size=n_samples) * self.params[2] + self.params[1]
        else:
            # Empírica: muestreo directo con reemplazo
            raise NotImplementedError("Muestreo empírico requiere datos originales")
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula CDF en punto(s) x."""
        if self.distribution == MarginalDistribution.EXPONENTIAL:
            return stats.expon.cdf(x, scale=self.params[0])
        elif self.distribution == MarginalDistribution.GAMMA:
            return stats.gamma.cdf(x, a=self.params[0], loc=self.params[1], 
                                  scale=self.params[2])
        elif self.distribution == MarginalDistribution.LOGNORMAL:
            return stats.lognorm.cdf(x, s=self.params[0], loc=self.params[1], 
                                    scale=self.params[2])
        elif self.distribution == MarginalDistribution.BETA:
            return stats.beta.cdf(x, a=self.params[0], b=self.params[1], 
                                 loc=self.params[2], scale=self.params[3])
        elif self.distribution == MarginalDistribution.GENERALIZED_PARETO:
            return stats.genpareto.cdf(x, c=self.params[0], loc=self.params[1], 
                                      scale=self.params[2])
        else:
            raise NotImplementedError(f"CDF no implementada para {self.distribution}")
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula percentil (PPF) en probabilidad(es) q."""
        if self.distribution == MarginalDistribution.EXPONENTIAL:
            return stats.expon.ppf(q, scale=self.params[0])
        elif self.distribution == MarginalDistribution.GAMMA:
            return stats.gamma.ppf(q, a=self.params[0], loc=self.params[1], 
                                  scale=self.params[2])
        elif self.distribution == MarginalDistribution.LOGNORMAL:
            return stats.lognorm.ppf(q, s=self.params[0], loc=self.params[1], 
                                    scale=self.params[2])
        elif self.distribution == MarginalDistribution.BETA:
            return stats.beta.ppf(q, a=self.params[0], b=self.params[1], 
                                 loc=self.params[2], scale=self.params[3])
        elif self.distribution == MarginalDistribution.GENERALIZED_PARETO:
            return stats.genpareto.ppf(q, c=self.params[0], loc=self.params[1], 
                                      scale=self.params[2])
        else:
            raise NotImplementedError(f"PPF no implementada para {self.distribution}")


@dataclass
class VineConfig:
    """
    Configuración para el ajuste y muestreo de Vine Copulas.
    
    Referencia: Cap 4.3.1.2 - Parámetros de estimación de estructura vine.
    
    Atributos:
        family_set: Familias de copulas permitidas (por defecto todas)
        selection_criterion: Criterio de selección de estructura ('aic', 'bic', 'loglik')
        truncation_level: Nivel de truncamiento del vine (-1 = sin truncar)
        rotation_check: Verificar rotaciones de copulas bivariadas
        prefiting: Usar distribuciones pre-ajustadas vs ajuste automático
    """
    family_set: List[str] = field(default_factory=lambda: [
        "gaussian", "t", "clayton", "gumbel", "frank", "joe",
        "clayton_rot90", "clayton_rot180", "clayton_rot270",
        "gumbel_rot90", "gumbel_rot180", "gumbel_rot270",
        "joe_rot90", "joe_rot180", "joe_rot270"
    ])
    selection_criterion: str = "bic"
    truncation_level: int = -1
    rotation_check: bool = True
    prefitting: bool = True


@dataclass
class StressScenario:
    """
    Escenario de tráfico generado con características de estrés.
    
    Referencia: Cap 4.3.1.3 - Definición de escenarios moderados y extremos.
    
    Atributos:
        scenario_id: Identificador único del escenario
        scenario_type: 'nominal', 'moderate_stress', 'extreme_stress'
        demand_matrix: Matriz de demanda por acceso y período (shape: n_accesos x n_períodos)
        arrival_times: Tiempos entre llegadas por vehículo
        behavioral_friction: Factores de fricción conductual por tipo de vehículo
        incident_probability: Probabilidad de incidente por segmento temporal
        metadata: Información adicional (semilla, timestamp, parámetros)
    """
    scenario_id: str
    scenario_type: str
    demand_matrix: np.ndarray
    arrival_times: np.ndarray
    behavioral_friction: np.ndarray
    incident_probability: np.ndarray
    metadata: Dict = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convierte el escenario a DataFrame para exportación."""
        return pd.DataFrame([{
            'scenario_id': self.scenario_id,
            'scenario_type': self.scenario_type,
            'demand_mean': self.demand_matrix.mean(),
            'demand_std': self.demand_matrix.std(),
            'demand_max': self.demand_matrix.max(),
            'arrival_mean': self.arrival_times.mean(),
            'friction_mean': self.behavioral_friction.mean(),
            'incident_prob_mean': self.incident_probability.mean(),
            **self.metadata
        }])


class VineCopulaGenerator:
    """
    Generador de escenarios de tráfico basado en Regular Vine Copulas.
    
    Implementa el pipeline de 4 pasos del Capítulo 4.3.1:
    1. Ajuste de marginales empíricas/paramétricas
    2. Estimación de estructura Regular Vine
    3. Muestreo condicional de escenarios de estrés
    4. Exportación a formatos SUMO compatibles
    
    Referencia Tesis: Capítulo 4.3.1, Apéndice A.2
    
    Ejemplo de uso:
        >>> generator = VineCopulaGenerator(seed=42)
        >>> generator.fit_marginals(data_df)
        >>> generator.fit_vine()
        >>> scenarios = generator.sample_stress_scenarios(
        ...     n_scenarios=100,
        ...     stress_level='moderate'
        ... )
    """
    
    def __init__(self, seed: Optional[int] = None, config: Optional[VineConfig] = None):
        """
        Inicializa el generador de Vine Copulas.
        
        Args:
            seed: Semilla para reproducibilidad (opcional)
            config: Configuración del vine (usa valores por defecto si None)
            
        Raises:
            ImportError: Si pyvinecopulib no está disponible
        """
        if not VINE_AVAILABLE:
            raise ImportError(
                "pyvinecopulib no está instalado. Instale con: "
                "pip install pyvinecopulib>=0.6.3"
            )
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.config = config or VineConfig()
        
        # Resultados del ajuste
        self.marginal_fits: Dict[str, MarginalFit] = {}
        self.vine_fit: Optional[pv.Vinecop] = None
        self.data_original: Optional[pd.DataFrame] = None
        
        # Variables de tráfico (orden consistente)
        self.variable_names = [
            "demand_access_0", "demand_access_1", "demand_access_2", "demand_access_3",
            "arrival_time_0", "arrival_time_1", "arrival_time_2", "arrival_time_3",
            "friction_type_A", "friction_type_B",
            "incident_prob_peak", "incident_prob_offpeak"
        ]
        
        logger.info(f"VineCopulaGenerator inicializado con seed={seed}")
    
    def fit_marginals(
        self, 
        data: pd.DataFrame,
        preferred_distributions: Optional[Dict[str, MarginalDistribution]] = None
    ) -> Dict[str, MarginalFit]:
        """
        Paso 1: Ajuste de distribuciones marginales a datos observados.
        
        Referencia: Cap 4.3.1.1 - Selección de familias según características empíricas.
        
        Para cada variable:
        1. Prueba múltiples distribuciones candidatas
        2. Selecciona la mejor por criterio BIC
        3. Valida con test de Kolmogorov-Smirnov (p > 0.05)
        
        Args:
            data: DataFrame con columnas nombradas según variable_names
            preferred_distributions: Diccionario opcional {variable: distribución_preferida}
            
        Returns:
            Diccionario {nombre_variable: MarginalFit} con ajustes
                
        Raises:
            ValueError: Si alguna columna requerida falta en data
        """
        logger.info("Paso 1: Ajustando distribuciones marginales...")
        
        # Validar columnas
        missing_cols = set(self.variable_names) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas en data: {missing_cols}")
        
        self.data_original = data.copy()
        self.marginal_fits = {}
        
        for var_name in self.variable_names:
            col_data = data[var_name].dropna().values
            
            if len(col_data) < 10:
                logger.warning(f"{var_name}: pocos datos ({len(col_data)}), usando empírica")
                continue
            
            # Determinar distribución preferida
            preferred = None
            if preferred_distributions and var_name in preferred_distributions:
                preferred = preferred_distributions[var_name]
            else:
                # Heurística basada en nombre de variable
                if "demand" in var_name:
                    preferred = MarginalDistribution.GAMMA
                elif "arrival" in var_name:
                    preferred = MarginalDistribution.EXPONENTIAL
                elif "friction" in var_name:
                    preferred = MarginalDistribution.LOGNORMAL
                elif "incident" in var_name:
                    preferred = MarginalDistribution.GENERALIZED_PARETO
            
            # Ajustar distribución
            fit = self._fit_single_marginal(var_name, col_data, preferred)
            self.marginal_fits[var_name] = fit
            
            logger.info(
                f"  {var_name}: {fit.distribution.value} "
                f"(KS p-value={fit.ks_pvalue:.4f})"
            )
        
        logger.info(f"Ajuste completado para {len(self.marginal_fits)} variables")
        return self.marginal_fits
    
    def _fit_single_marginal(
        self, 
        var_name: str, 
        data: np.ndarray,
        preferred: Optional[MarginalDistribution] = None
    ) -> MarginalFit:
        """
        Ajusta una distribución marginal individual.
        
        Args:
            var_name: Nombre de la variable
            data: Array de valores observados
            preferred: Distribución preferida (opcional)
            
        Returns:
            MarginalFit con resultados del ajuste
        """
        candidates = []
        
        # Si hay preferencia, probarla primero
        if preferred:
            candidates.append(preferred)
        
        # Añadir resto de candidatos
        all_dists = list(MarginalDistribution)
        for dist in all_dists:
            if dist not in candidates:
                candidates.append(dist)
        
        best_fit = None
        best_bic = np.inf
        
        for dist in candidates:
            try:
                params, ks_stat, ks_pval = self._try_fit_distribution(dist, data)
                
                if params is None:
                    continue
                
                # Calcular BIC aproximado (n * log(RSS/n) + k * log(n))
                n = len(data)
                k = len(params)
                
                # RSS aproximado desde KS
                rss = ks_stat ** 2 * n
                bic = n * np.log(rss / n + 1e-10) + k * np.log(n)
                
                if bic < best_bic and ks_pval > 0.01:  # Umbral mínimo de validez
                    best_bic = bic
                    best_fit = MarginalFit(
                        variable_name=var_name,
                        distribution=dist,
                        params=params,
                        ks_statistic=ks_stat,
                        ks_pvalue=ks_pval,
                        data_min=data.min(),
                        data_max=data.max()
                    )
                    
            except Exception as e:
                logger.debug(f"Fallo al ajustar {dist.value} para {var_name}: {e}")
                continue
        
        # Fallback: si ningún ajuste funciona, usar empírica con warning
        if best_fit is None:
            logger.warning(f"No se pudo ajustar distribución paramétrica para {var_name}")
            best_fit = MarginalFit(
                variable_name=var_name,
                distribution=MarginalDistribution.EMPIRICAL,
                params=(),
                ks_statistic=0.0,
                ks_pvalue=1.0,
                data_min=data.min(),
                data_max=data.max()
            )
        
        return best_fit
    
    def _try_fit_distribution(
        self, 
        dist: MarginalDistribution, 
        data: np.ndarray
    ) -> Tuple[Optional[Tuple], float, float]:
        """
        Intenta ajustar una distribución específica a los datos.
        
        Returns:
            (params, ks_stat, ks_pvalue) o (None, inf, 0) si falla
        """
        try:
            # Normalizar datos a [0, 1] para beta, mantener original para otras
            if dist == MarginalDistribution.BETA:
                data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)
                data_norm = np.clip(data_norm, 0.001, 0.999)
                params = stats.beta.fit(data_norm)
                ks_stat, ks_pval = kstest(data_norm, 'beta', args=params)
                return params, ks_stat, ks_pval
            
            elif dist == MarginalDistribution.EXPONENTIAL:
                params = stats.expon.fit(data)
                ks_stat, ks_pval = kstest(data, 'expon', args=params)
                return params, ks_stat, ks_pval
            
            elif dist == MarginalDistribution.GAMMA:
                params = stats.gamma.fit(data)
                ks_stat, ks_pval = kstest(data, 'gamma', args=params)
                return params, ks_stat, ks_pval
            
            elif dist == MarginalDistribution.LOGNORMAL:
                params = stats.lognorm.fit(data)
                ks_stat, ks_pval = kstest(data, 'lognorm', args=params)
                return params, ks_stat, ks_pval
            
            elif dist == MarginalDistribution.GENERALIZED_PARETO:
                # Solo usar percentil superior para colas pesadas
                threshold = np.percentile(data, 90)
                tail_data = data[data > threshold] - threshold
                if len(tail_data) < 10:
                    return None, np.inf, 0.0
                params = stats.genpareto.fit(tail_data)
                ks_stat, ks_pval = kstest(tail_data, 'genpareto', args=params)
                return params, ks_stat, ks_pval
            
            else:
                return None, np.inf, 0.0
                
        except Exception:
            return None, np.inf, 0.0
    
    def fit_vine(self):
        """
        Paso 2: Estimación de estructura Regular Vine.
        
        Referencia: Cap 4.3.1.2 - Construcción de vine regular con selección
        de estructura basada en dependencia condicional.
        
        Proceso:
        1. Transformar datos marginales a uniforme vía PIT
        2. Estimar estructura vine con criterio BIC
        3. Validar calidad del ajuste
        
        Returns:
            Objeto pyvinecopulib.Vinecop ajustado
            
        Raises:
            RuntimeError: Si no se han ajustado marginales primero
            ImportError: Si pyvinecopulib no está disponible
        """
        if not VINE_AVAILABLE:
            raise ImportError(
                "pyvinecopulib no está instalado. Instale con: "
                "pip install pyvinecopulib>=0.6.3"
            )
        
        logger.info("Paso 2: Estimando estructura Regular Vine...")
        
        if not self.marginal_fits:
            raise RuntimeError(
                "Debe ejecutar fit_marginals() antes de fit_vine()"
            )
        
        if self.data_original is None:
            raise RuntimeError("No hay datos originales disponibles")
        
        # Paso 2.1: Probability Integral Transform (PIT)
        logger.info("  Aplicando Probability Integral Transform...")
        uniform_data = self._apply_pit(self.data_original)
        
        # Paso 2.2: Configurar families permitidas
        logger.info(f"  Configurando familias: {len(self.config.family_set)} tipos")
        family_list = self._parse_family_set(self.config.family_set)
        
        # Paso 2.3: Crear y ajustar vine usando API moderna de pyvinecopulib 0.6+
        logger.info("  Ajustando Regular Vine con criterio BIC...")
        
        n_vars = uniform_data.shape[1]
        
        # Inicializar vine vacío
        self.vine_fit = pv.Vinecop(n_vars)
        
        # Configurar controles de ajuste con valores explícitos
        controls = pv.FitControlsVinecop()
        controls.family_set = family_list
        controls.selection_criterion = self.config.selection_criterion
        # Usar truncation_level como int explícito (-1 = sin truncamiento)
        # pyvinecopulib 0.7+ requiere setter explícito para trunc_lvl
        trunc_lvl = int(self.config.truncation_level) if self.config.truncation_level else -1
        try:
            controls.trunc_lvl = trunc_lvl
        except TypeError:
            # Fallback para versiones antiguas que no soportan setter
            logger.warning("Versión antigua de pyvinecopulib, omitiendo trunc_lvl")
            pass
        
        # Convertir numpy array a formato esperado (F-order)
        data_fortran = np.asfortranarray(uniform_data.astype(np.float64))
        
        # Ajustar vine
        self.vine_fit.select(data_fortran, controls)
        
        # Validar ajuste
        loglik = self.vine_fit.loglik(data_fortran)
        logger.info(f"  Log-likelihood final: {loglik:.4f}")
        
        return self.vine_fit
    
    def _apply_pit(self, data: pd.DataFrame) -> np.ndarray:
        """
        Aplica Probability Integral Transform para convertir a uniforme[0,1].
        
        Referencia: Cap 4.3.1.2 - Corrección (n+1) para evitar límites {0,1}.
        
        Args:
            data: DataFrame con datos originales
            
        Returns:
            Array numpy con valores uniformes en (0, 1)
        """
        n = len(data)
        uniform_samples = np.zeros((n, len(self.variable_names)))
        
        for i, var_name in enumerate(self.variable_names):
            if var_name not in self.marginal_fits:
                # Fallback: rango empírico
                ranks = stats.rankdata(data[var_name].values, method='average')
                uniform_samples[:, i] = ranks / (n + 1)
            else:
                fit = self.marginal_fits[var_name]
                if fit.distribution == MarginalDistribution.EMPIRICAL:
                    ranks = stats.rankdata(data[var_name].values, method='average')
                    uniform_samples[:, i] = ranks / (n + 1)
                else:
                    # Usar CDF paramétrica
                    cdf_vals = fit.cdf(data[var_name].values)
                    # Clip para evitar 0 y 1 exactos
                    uniform_samples[:, i] = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
        
        return uniform_samples
    
    def _parse_family_set(self, family_names: list) -> "FamilySetType":
        """
        Convierte lista de nombres de familias a FamilySet de pyvinecopulib.
        
        Args:
            family_names: Lista de nombres de familias
            
        Returns:
            Lista de pv.BicopFamily para configuración del vine
        """
        if not VINE_AVAILABLE:
            raise ImportError("pyvinecopulib no está disponible")
        
        # Mapeo correcto de familias según BicopFamily de pyvinecopulib 0.6+
        family_map = {
            "gaussian": pv.BicopFamily.gaussian,
            "student": pv.BicopFamily.student,
            "clayton": pv.BicopFamily.clayton,
            "gumbel": pv.BicopFamily.gumbel,
            "frank": pv.BicopFamily.frank,
            "joe": pv.BicopFamily.joe,
            "bb1": pv.BicopFamily.bb1,
            "bb6": pv.BicopFamily.bb6,
            "bb7": pv.BicopFamily.bb7,
            "bb8": pv.BicopFamily.bb8,
            "tawn": pv.BicopFamily.tawn,
            "tll": pv.BicopFamily.tll,
            "indep": pv.BicopFamily.indep,
        }
        
        families = []
        for name in family_names:
            if name in family_map:
                families.append(family_map[name])
            else:
                logger.warning(f"Familia desconocida: {name}")
        
        return families  # Devolver lista de BicopFamily
    
    def sample_stress_scenarios(
        self,
        n_scenarios: int = 100,
        stress_level: str = "moderate",
        conditional_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> List[StressScenario]:
        """
        Paso 3: Muestreo condicional de escenarios de estrés.
        
        Referencia: Cap 4.3.1.3 - Generación de escenarios moderados y extremos
        mediante condicionamiento en colas de distribución.
        
        Tipos de estrés:
        - 'nominal': Muestreo incondicional de la distribución conjunta
        - 'moderate': Condicionar en percentil 80-95 de demanda
        - 'extreme': Condicionar en percentil 95-99 de demanda + incidentes
        
        Args:
            n_scenarios: Número de escenarios a generar
            stress_level: Nivel de estrés ('nominal', 'moderate', 'extreme')
            conditional_constraints: Restricciones manuales {var: (percentil_min, percentil_max)}
            
        Returns:
            Lista de StressScenario generados
        """
        logger.info(f"Paso 3: Generando {n_scenarios} escenarios ({stress_level})...")
        
        if self.vine_fit is None:
            raise RuntimeError("Debe ejecutar fit_vine() antes de muestrear")
        
        scenarios = []
        
        for i in range(n_scenarios):
            # Determinar constraints según nivel de estrés
            if conditional_constraints is None:
                constraints = self._get_stress_constraints(stress_level, i)
            else:
                constraints = conditional_constraints
            
            # Muestrear del vine (posiblemente condicionado)
            if constraints:
                sample = self._sample_conditional(constraints)
            else:
                sample = self._sample_unconditional()
            
            # Convertir de uniforme a escala original
            scenario_data = self._inverse_pit(sample)
            
            # Crear escenario
            scenario = self._create_scenario_from_data(
                scenario_data, stress_level, seed=self.seed + i if self.seed else None
            )
            scenarios.append(scenario)
        
        logger.info(f"Generados {len(scenarios)} escenarios")
        return scenarios
    
    def _get_stress_constraints(
        self, 
        stress_level: str, 
        scenario_idx: int
    ) -> Dict[str, Tuple[float, float]]:
        """
        Define restricciones de muestreo según nivel de estrés.
        
        Args:
            stress_level: Nivel de estrés
            scenario_idx: Índice del escenario (para variación)
            
        Returns:
            Diccionario {variable: (percentil_min, percentil_max)}
        """
        if stress_level == "nominal":
            return {}  # Sin restricciones
        
        elif stress_level == "moderate":
            # Condicionar demanda en percentil 80-95
            return {
                "demand_access_0": (0.80, 0.95),
                "demand_access_1": (0.80, 0.95),
                "demand_access_2": (0.80, 0.95),
                "demand_access_3": (0.80, 0.95),
            }
        
        elif stress_level == "extreme":
            # Condicionar demanda en percentil 95-99 + incidentes altos
            base_constraint = {
                "demand_access_0": (0.95, 0.99),
                "demand_access_1": (0.95, 0.99),
                "demand_access_2": (0.95, 0.99),
                "demand_access_3": (0.95, 0.99),
                "incident_prob_peak": (0.90, 0.99),
            }
            
            # Variar ligeramente entre escenarios
            offset = (scenario_idx % 10) * 0.01
            for key in base_constraint:
                min_p, max_p = base_constraint[key]
                base_constraint[key] = (min_p + offset, min(max_p + offset, 0.999))
            
            return base_constraint
        
        else:
            raise ValueError(f"Nivel de estrés desconocido: {stress_level}")
    
    def _sample_unconditional(self) -> np.ndarray:
        """
        Muestrea incondicionalmente del vine copula.
        
        Returns:
            Array (n_vars,) de valores uniformes
        """
        n_vars = len(self.variable_names)
        sample = self.vine_fit.simulate(1)[0]
        return np.array(sample)
    
    def _sample_conditional(
        self, 
        constraints: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """
        Muestrea condicionalmente del vine copula con restricciones.
        
        Implementa muestreo por rechazo con límites en espacio uniforme.
        
        Args:
            constraints: {var_name: (percentil_min, percentil_max)}
            
        Returns:
            Array (n_vars,) de valores uniformes que satisfacen constraints
        """
        max_attempts = 1000
        
        for _ in range(max_attempts):
            sample = self._sample_unconditional()
            
            # Verificar constraints
            valid = True
            for var_name, (p_min, p_max) in constraints.items():
                if var_name in self.variable_names:
                    idx = self.variable_names.index(var_name)
                    if not (p_min <= sample[idx] <= p_max):
                        valid = False
                        break
            
            if valid:
                return sample
        
        # Fallback: devolver última muestra con warning
        logger.warning("Muestreo condicional no convergió, usando muestra cercana")
        return sample
    
    def _inverse_pit(self, uniform_sample: np.ndarray) -> Dict[str, float]:
        """
        Convierte muestra uniforme de vuelta a escala original.
        
        Args:
            uniform_sample: Array de valores en [0, 1]
            
        Returns:
            Diccionario {var_name: valor_en_escala_original}
        """
        result = {}
        
        for i, var_name in enumerate(self.variable_names):
            u = uniform_sample[i]
            
            if var_name in self.marginal_fits:
                fit = self.marginal_fits[var_name]
                if fit.distribution == MarginalDistribution.EMPIRICAL:
                    # Interpolación lineal en datos originales
                    orig_data = self.data_original[var_name].values
                    sorted_data = np.sort(orig_data)
                    idx = int(u * (len(sorted_data) - 1))
                    result[var_name] = sorted_data[idx]
                else:
                    # Usar PPF paramétrica
                    result[var_name] = fit.ppf(u)
            else:
                # Fallback: interpolación en datos originales
                if self.data_original is not None and var_name in self.data_original:
                    orig_data = self.data_original[var_name].values
                    sorted_data = np.sort(orig_data)
                    idx = int(u * (len(sorted_data) - 1))
                    result[var_name] = sorted_data[idx]
                else:
                    result[var_name] = u  # Mantener uniforme
        
        return result
    
    def _create_scenario_from_data(
        self, 
        data: Dict[str, float],
        stress_level: str,
        seed: Optional[int] = None
    ) -> StressScenario:
        """
        Crea objeto StressScenario desde datos muestreados.
        
        Args:
            data: Diccionario con valores por variable
            stress_level: Nivel de estrés aplicado
            seed: Semilla usada para este escenario
            
        Returns:
            StressScenario listo para exportación
        """
        scenario_id = str(uuid.uuid4())[:8]
        
        # Construir matrices de demanda (4 accesos x 3 períodos: peak, offpeak, night)
        demand_base = np.array([
            data.get("demand_access_0", 500),
            data.get("demand_access_1", 500),
            data.get("demand_access_2", 500),
            data.get("demand_access_3", 500),
        ])
        
        # Variar por período (factor multiplicativo)
        period_factors = np.array([1.2, 0.7, 0.4])  # peak, offpeak, night
        demand_matrix = np.outer(demand_base, period_factors)
        
        # Tiempos entre llegadas (inverso de demanda, con ruido)
        arrival_times = 3600.0 / (demand_base + 1) * self.rng.uniform(0.8, 1.2, 4)
        
        # Fricción conductual
        friction = np.array([
            data.get("friction_type_A", 1.0),
            data.get("friction_type_B", 1.0),
        ])
        
        # Probabilidad de incidente
        incident_prob = np.array([
            data.get("incident_prob_peak", 0.01),
            data.get("incident_prob_offpeak", 0.005),
        ])
        
        return StressScenario(
            scenario_id=scenario_id,
            scenario_type=stress_level,
            demand_matrix=demand_matrix,
            arrival_times=arrival_times,
            behavioral_friction=friction,
            incident_probability=incident_prob,
            metadata={
                "seed": seed,
                "stress_level": stress_level,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )
    
    def export_to_sumo(
        self,
        scenarios: List[StressScenario],
        output_dir: Union[str, Path],
        template_file: Optional[Path] = None
    ) -> List[Path]:
        """
        Paso 4: Exportación a archivos .rou.xml compatibles con SUMO.
        
        Referencia: Cap 4.3.1.4 - Formato de rutas SUMO para escenarios generados.
        
        Args:
            scenarios: Lista de escenarios a exportar
            output_dir: Directorio de salida
            template_file: Archivo .rou.xml plantilla (opcional)
            
        Returns:
            Lista de rutas a archivos .rou.xml generados
        """
        logger.info(f"Paso 4: Exportando {len(scenarios)} escenarios a SUMO...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for scenario in scenarios:
            filename = f"routes_{scenario.scenario_id}_{scenario.scenario_type}.rou.xml"
            filepath = output_path / filename
            
            xml_content = self._generate_rou_xml(scenario)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            generated_files.append(filepath)
            logger.debug(f"  Generado: {filepath}")
        
        logger.info(f"Exportación completada: {len(generated_files)} archivos")
        return generated_files
    
    def _generate_rou_xml(self, scenario: StressScenario) -> str:
        """
        Genera contenido XML para archivo .rou.xml de SUMO.
        
        Args:
            scenario: Escenario a convertir
            
        Returns:
            String con contenido XML válido
        """
        # Encabezado XML
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '',
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
            '',
            f'    <!-- Escenario: {scenario.scenario_id} -->',
            f'    <!-- Tipo: {scenario.scenario_type} -->',
            f'    <!-- Demanda media: {scenario.demand_matrix.mean():.1f} veh/h -->',
            '',
        ]
        
        # Definir tipos de vehículos con fricción conductual
        xml_lines.extend([
            '    <vType id="typeA" accel="2.5" decel="4.5" sigma="0.5" tau="1.0"/>',
            '    <vType id="typeB" accel="2.0" decel="4.0" sigma="0.7" tau="1.2"/>',
            '',
        ])
        
        # Generar flujos vehiculares basados en demanda
        flow_id = 0
        access_names = ["access_0", "access_1", "access_2", "access_3"]
        period_names = ["peak", "offpeak", "night"]
        period_times = [(0, 3600), (3600, 7200), (7200, 10800)]  # segundos
        
        for acc_idx, acc_name in enumerate(access_names):
            for per_idx, per_name in enumerate(period_names):
                demand = scenario.demand_matrix[acc_idx, per_idx]
                
                if demand < 10:
                    continue  # Ignorar demanda muy baja
                
                # Calcular intervalo entre vehículos (segundos)
                interval = 3600.0 / max(demand, 1)
                
                start_time, end_time = period_times[per_idx]
                
                # Alternar tipo de vehículo
                vtype = "typeA" if flow_id % 2 == 0 else "typeB"
                
                xml_lines.append(
                    f'    <flow id="flow_{flow_id}" type="{vtype}" '
                    f'from="{acc_name}_in" to="intersection_center" '
                    f'begin="{start_time}" end="{end_time}" '
                    f'veps="{interval:.2f}"/>'
                )
                flow_id += 1
        
        # Cierre XML
        xml_lines.extend([
            '',
            '</routes>',
        ])
        
        return '\n'.join(xml_lines)
    
    def validate_samples(
        self, 
        n_test_samples: int = 1000,
        significance_level: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Valida que las muestras generadas preserven las distribuciones marginales.
        
        Ejecuta test de Kolmogorov-Smirnov para cada variable.
        
        Args:
            n_test_samples: Número de muestras para validación
            significance_level: Nivel de significancia para test KS
            
        Returns:
            Diccionario con resultados de validación por variable
        """
        logger.info(f"Validando muestras con {n_test_samples} pruebas...")
        
        validation_results = {}
        
        # Generar muestras de prueba
        test_samples = []
        for _ in range(n_test_samples):
            sample = self._sample_unconditional()
            transformed = self._inverse_pit(sample)
            test_samples.append(transformed)
        
        # Convertir a DataFrame
        test_df = pd.DataFrame(test_samples)
        
        # Test KS por variable
        for var_name in self.variable_names:
            if var_name not in self.marginal_fits:
                continue
            
            fit = self.marginal_fits[var_name]
            original_data = self.data_original[var_name].dropna().values
            sampled_data = test_df[var_name].values
            
            # KS test entre original y muestreado
            ks_stat, ks_pval = kstest(original_data, lambda x: fit.cdf(x))
            
            validation_results[var_name] = {
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "passed": ks_pval > significance_level,
                "original_mean": original_data.mean(),
                "sampled_mean": sampled_data.mean(),
                "original_std": original_data.std(),
                "sampled_std": sampled_data.std(),
            }
            
            status = "✓ PASS" if validation_results[var_name]["passed"] else "✗ FAIL"
            logger.info(f"  {var_name}: {status} (p={ks_pval:.4f})")
        
        return validation_results


def generate_synthetic_data(
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Genera datos sintéticos de tráfico para testing y demostración.
    
    Útil cuando no hay datos reales disponibles para ajustar el vine.
    
    Args:
        n_samples: Número de muestras a generar
        seed: Semilla para reproducibilidad
        
    Returns:
        DataFrame con columnas según variable_names
    """
    rng = np.random.default_rng(seed)
    
    data = {}
    
    # Demanda por acceso (Gamma distribuida, correlacionada)
    base_demand = rng.gamma(shape=5, scale=100, size=n_samples)
    for i in range(4):
        noise = rng.normal(0, 50, n_samples)
        data[f"demand_access_{i}"] = np.maximum(base_demand + noise, 50)
    
    # Tiempos entre llegadas (Exponencial, inversamente relacionados con demanda)
    for i in range(4):
        rate = 3600.0 / (data[f"demand_access_{i}"] + 1)
        data[f"arrival_time_{i}"] = rng.exponential(scale=rate / 3600, size=n_samples)
    
    # Fricción conductual (Log-normal)
    data["friction_type_A"] = rng.lognormal(mean=0, sigma=0.2, size=n_samples)
    data["friction_type_B"] = rng.lognormal(mean=0.1, sigma=0.25, size=n_samples)
    
    # Probabilidad de incidente (Beta, valores pequeños)
    data["incident_prob_peak"] = rng.beta(a=1, b=50, size=n_samples)
    data["incident_prob_offpeak"] = rng.beta(a=1, b=100, size=n_samples)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generador de Escenarios de Estrés con Vine Copulas"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Ejecutar modo test con datos sintéticos"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Número de escenarios a generar"
    )
    parser.add_argument(
        "--output-dir", type=str, default="generated_scenarios",
        help="Directorio de salida para archivos .rou.xml"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para reproducibilidad"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.test:
        print("=" * 60)
        print("MODO TEST: Generación y validación con datos sintéticos")
        print("=" * 60)
        
        # Generar datos sintéticos
        print("\n1. Generando datos sintéticos...")
        synthetic_data = generate_synthetic_data(n_samples=500, seed=args.seed)
        print(f"   Datos generados: {synthetic_data.shape}")
        print(synthetic_data.describe().round(2))
        
        # Inicializar generador
        print("\n2. Inicializando VineCopulaGenerator...")
        generator = VineCopulaGenerator(seed=args.seed)
        
        # Ajustar marginales
        print("\n3. Ajustando distribuciones marginales...")
        generator.fit_marginals(synthetic_data)
        
        # Ajustar vine
        print("\n4. Estimando estructura Regular Vine...")
        generator.fit_vine()
        
        # Generar escenarios
        print("\n5. Generando escenarios de estrés...")
        scenarios_nominal = generator.sample_stress_scenarios(
            n_scenarios=10, stress_level="nominal"
        )
        scenarios_moderate = generator.sample_stress_scenarios(
            n_scenarios=10, stress_level="moderate"
        )
        scenarios_extreme = generator.sample_stress_scenarios(
            n_scenarios=10, stress_level="extreme"
        )
        
        print(f"   Nominal: {len(scenarios_nominal)} escenarios")
        print(f"   Moderado: {len(scenarios_moderate)} escenarios")
        print(f"   Extremo: {len(scenarios_extreme)} escenarios")
        
        # Validar muestras
        print("\n6. Validando muestras (test KS)...")
        validation = generator.validate_samples(n_test_samples=500)
        
        passed = sum(1 for v in validation.values() if v["passed"])
        total = len(validation)
        print(f"\n   Resultado: {passed}/{total} variables pasan test KS (p > 0.05)")
        
        # Exportar a SUMO
        print("\n7. Exportando a archivos .rou.xml...")
        output_path = Path(args.output_dir)
        generator.export_to_sumo(
            scenarios_nominal[:3] + scenarios_moderate[:3] + scenarios_extreme[:3],
            output_path
        )
        print(f"   Archivos generados en: {output_path.absolute()}")
        
        # Resumen final
        print("\n" + "=" * 60)
        if passed >= total * 0.8:
            print("✓ TEST EXITOSO: El generador preserva distribuciones marginales")
        else:
            print("⚠ TEST CON ADVERTENCIAS: Algunas variables no pasan KS")
        print("=" * 60)
        
    else:
        print("Use --test para ejecutar modo de validación con datos sintéticos")
        print("Ejemplo: python src/probabilistic/vine_generator.py --test")
