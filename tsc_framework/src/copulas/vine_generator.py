"""
vine_generator.py
=================
Módulo 2: Generador Probabilístico de Escenarios de Estrés

Utiliza una Vine Copula (C-Vine / D-Vine) para capturar la estructura de
dependencia entre tres variables de tráfico:

    X₁ = Demand        (vehículos/intervalo — datos macro)
    X₂ = Accel_Mag     (magnitud de aceleración — datos micro)
    X₃ = Jerk          (derivada de aceleración — datos micro)

Flujo matemático:
─────────────────
1. Carga de datos observados  →  [Demand, Accel_Mag, Jerk]
2. Probability Integral Transform (PIT):
       uᵢ = F̂(xᵢ) = rank(xᵢ) / (n + 1)     ∀ i ∈ {1,2,3}
   Transforma cada variable al dominio [0, 1] usando la ECDF empírica.
3. Ajuste de la Vine Copula C(u₁, u₂, u₃) con pyvinecopulib.
4. Simulación Monte Carlo: n muestras uniformes del modelo ajustado.
5. Transformación inversa (Quantile Function):
       x̃ᵢ = F̂⁻¹(ũᵢ) = Percentil(ũᵢ · 100, datos originales)
   Proyecta las muestras uniformes de vuelta al dominio físico original.
6. Exportación a data/processed/stress_scenarios.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# pyvinecopulib es el motor de Vine Copulas
try:
    import pyvinecopulib as pv
    _PV_AVAILABLE = True
except ImportError:
    _PV_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MACRO_CSV = PROCESSED_DIR / "macro_demand.csv"
MICRO_CSV = PROCESSED_DIR / "micro_behavior.csv"
SCENARIOS_CSV = PROCESSED_DIR / "stress_scenarios.csv"
COPULA_SUMMARY_JSON = PROCESSED_DIR / "vine_copula_summary.json"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VARIABLE_NAMES: list[str] = ["demand", "accel_magnitude", "jerk"]
N_STRESS_SAMPLES: int = 2_000       # Escenarios sintéticos a generar
RANDOM_SEED: int = 42


# ===========================================================================
# Estructura de datos tipada para el resultado
# ===========================================================================

class VineGeneratorResult(NamedTuple):
    """Contenedor tipado con los artefactos generados por el pipeline."""
    joint_data: pd.DataFrame          # Datos conjuntos antes de la PIT
    pseudo_obs: np.ndarray            # Pseudo-observaciones u ∈ [0,1]³
    model: object                     # Objeto Vinecop ajustado
    scenarios_df: pd.DataFrame        # Escenarios sintéticos en dominio real
    summary: dict                     # Resumen del modelo para la tesis


# ===========================================================================
# PASO 1 — CARGA Y CONSTRUCCIÓN DEL DATASET CONJUNTO
# ===========================================================================

def load_joint_data(
    macro_path: Path = MACRO_CSV,
    micro_path: Path = MICRO_CSV,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Carga los datos procesados y construye un dataset conjunto alineando
    temporalmente los datos macro (agrupados) y micro (telemetría).

    Estrategia de alineación (Bootstrap / Remuestreo):
    ────────────────────────────────────────────────────
    Dado que los datos macro tienen granularidad de 15 min y los micro
    son telemetría a 1 Hz, tienen longitudes incompatibles. Para construir
    la matriz de entrenamiento de la cópula se usa remuestreo con reemplazo
    (bootstrap) de las variables micro, ajustándolas a la longitud de la
    serie macro. Esto preserva la distribución marginal de cada variable.

    Args:
        macro_path: Ruta a macro_demand.csv.
        micro_path: Ruta a micro_behavior.csv.
        random_state: Semilla para reproducibilidad del muestreo.

    Returns:
        DataFrame con columnas [demand, accel_magnitude, jerk].

    Raises:
        FileNotFoundError: Si alguno de los CSV no existe.
        ValueError: Si las columnas requeridas no están presentes.
    """
    logger.info("─" * 60)
    logger.info("PASO 1 — Cargando y construyendo dataset conjunto")
    logger.info("─" * 60)

    # ── Validar existencia de archivos ──────────────────────────────────────
    for path in (macro_path, micro_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {path}\n"
                f"Ejecuta primero: python src/data_pipeline/process_data.py"
            )

    # ── Cargar datos macro ──────────────────────────────────────────────────
    logger.info(f"Leyendo macro: {macro_path.name}")
    macro_df = pd.read_csv(macro_path)

    if "vehicle_count" not in macro_df.columns:
        raise ValueError(
            f"Columna 'vehicle_count' no encontrada en {macro_path.name}.\n"
            f"Columnas disponibles: {list(macro_df.columns)}"
        )

    demand_series = pd.to_numeric(macro_df["vehicle_count"], errors="coerce").dropna()
    logger.info(f"  Demanda: {len(demand_series):,} bins | "
                f"rango [{demand_series.min():.0f}, {demand_series.max():.0f}] veh/intervalo")

    # ── Cargar datos micro ──────────────────────────────────────────────────
    logger.info(f"Leyendo micro: {micro_path.name}")
    micro_df = pd.read_csv(micro_path, low_memory=False)

    # Columnas requeridas (nombres reales del pipeline de procesamiento)
    required_micro = ["accel_magnitude", "jerk"]
    missing = [c for c in required_micro if c not in micro_df.columns]
    if missing:
        raise ValueError(
            f"Columnas requeridas no encontradas en {micro_path.name}: {missing}\n"
            f"Columnas disponibles: {list(micro_df.columns)}"
        )

    accel_series = pd.to_numeric(micro_df["accel_magnitude"], errors="coerce").dropna()
    jerk_series = pd.to_numeric(micro_df["jerk"], errors="coerce").dropna()

    logger.info(f"  Aceleración: {len(accel_series):,} muestras | "
                f"media={accel_series.mean():.4f} m/s²")
    logger.info(f"  Jerk:        {len(jerk_series):,} muestras | "
                f"media={jerk_series.mean():.4f} m/s³")

    # ── Bootstrap: alinear longitudes ──────────────────────────────────────
    # La serie más corta es la de referencia (demand_series es generalmente
    # la más corta al ser series temporales agregadas)
    n_ref = len(demand_series)
    rng = np.random.default_rng(random_state)

    logger.info(f"Bootstrap: remuestreando micro a n={n_ref:,} (longitud de demand_series)")

    accel_boot = rng.choice(accel_series.values, size=n_ref, replace=True)
    jerk_boot = rng.choice(jerk_series.values, size=n_ref, replace=True)

    joint_df = pd.DataFrame({
        "demand":         demand_series.values[:n_ref],
        "accel_magnitude": accel_boot,
        "jerk":           jerk_boot,
    })

    # Eliminar filas con cualquier NaN residual
    joint_df = joint_df.dropna().reset_index(drop=True)
    logger.info(f"✓ Dataset conjunto: {len(joint_df):,} observaciones × {joint_df.shape[1]} variables")

    return joint_df


# ===========================================================================
# PASO 2 — PROBABILITY INTEGRAL TRANSFORM (PIT)
# ===========================================================================

def to_pseudo_observations(data: pd.DataFrame) -> np.ndarray:
    """
    Transforma las variables observadas al dominio [0, 1] usando la
    Función de Distribución Empírica (ECDF) — Probability Integral Transform.

    Fórmula:
        uᵢⱼ = rank(xᵢⱼ) / (n + 1)

    La corrección (n+1) en lugar de n evita valores en los límites exactos
    {0, 1}, lo que causaría problemas numéricos en el log de la función
    de verosimilitud de la cópula (log(0) → -∞).

    Equivalencia con la ECDF:
        uᵢ = F̂ₙ(xᵢ) donde F̂ₙ es el estimador de Kaplan-Meier discreto.

    Args:
        data: DataFrame con columnas numéricas a transformar.

    Returns:
        Array numpy de shape (n, d) con valores en (0, 1).
    """
    logger.info("─" * 60)
    logger.info("PASO 2 — Probability Integral Transform (PIT) → dominio [0,1]")
    logger.info("─" * 60)

    n = len(data)
    pseudo = np.empty((n, data.shape[1]), dtype=np.float64)

    for j, col in enumerate(data.columns):
        # rankdata usa método 'average' para empates por defecto
        pseudo[:, j] = rankdata(data[col].values) / (n + 1)
        logger.info(f"  u_{col}: rango [{pseudo[:, j].min():.6f}, {pseudo[:, j].max():.6f}]")

    # Verificar que todos los valores están estrictamente en (0, 1)
    assert (pseudo > 0).all() and (pseudo < 1).all(), \
        "¡Error PIT! Existen valores fuera del dominio (0, 1)."

    logger.info(f"✓ Pseudo-observaciones: shape {pseudo.shape} — dtype {pseudo.dtype}")
    return pseudo


# ===========================================================================
# PASO 3 — AJUSTE DE LA VINE COPULA
# ===========================================================================

def fit_vine_copula(pseudo_obs: np.ndarray) -> object:
    """
    Ajusta una Vine Copula (C-Vine o D-Vine) a las pseudo-observaciones
    usando máxima verosimilitud secuencial (método de árboles de cópulas pares).

    Fundamento matemático:
    ─────────────────────
    Una Vine Copula factoriza la densidad conjunta como:
        c(u₁, u₂, u₃) = ∏ᵢ ∏ⱼ cₑ(F(uᵢ|v), F(uⱼ|v))
    donde cada cₑ es una cópula par bivariante en un árbol T de la vina.

    pyvinecopulib selecciona automáticamente:
        - La estructura del árbol (orden de las variables)
        - La familia de cópula par (Gaussian, Clayton, Gumbel, Frank, etc.)
        - Los parámetros de cada cópula par por AIC/BIC

    Args:
        pseudo_obs: Array (n, d) con valores en (0, 1).

    Returns:
        Objeto Vinecop ajustado de pyvinecopulib.

    Raises:
        ImportError: Si pyvinecopulib no está instalado.
        RuntimeError: Si el ajuste falla.
    """
    logger.info("─" * 60)
    logger.info("PASO 3 — Ajustando Vine Copula (selección automática de estructura)")
    logger.info("─" * 60)

    if not _PV_AVAILABLE:
        raise ImportError(
            "pyvinecopulib no está instalado.\n"
            "Instala con: pip install pyvinecopulib"
        )

    # Garantizar dtype float64 estricto (requerido por pyvinecopulib)
    data_f64 = pseudo_obs.astype(np.float64)

    # Familias de cópulas a considerar: elípticas + arquimedianas.
    # En pyvinecopulib moderno (Python), FamilySet fue eliminado;
    # se pasa una lista de BicopFamily directamente a FitControlsVinecop.
    # Elliptic:     Gaussian, Student-t
    # Archimedean:  Clayton, Gumbel, Frank, Joe
    _families = [
        pv.BicopFamily.gaussian,
        pv.BicopFamily.student,
        pv.BicopFamily.clayton,
        pv.BicopFamily.gumbel,
        pv.BicopFamily.frank,
        pv.BicopFamily.joe,
    ]

    controls = pv.FitControlsVinecop(
        family_set=_families,
        selection_criterion="aic",   # Criterio de info de Akaike (penaliza complejidad)
        num_threads=1,               # Reproducibilidad: 1 hilo
    )

    try:
        logger.info("  Iniciando ajuste MLE secuencial de la Vine Copula...")
        # API actual: Vinecop(d) crea la estructura vacía; .select() realiza
        # la selección automática de familias y parámetros por MLE secuencial.
        dimension = data_f64.shape[1]
        cop = pv.Vinecop(d=dimension)
        cop.select(data=data_f64, controls=controls)
        logger.info(f"✓ Vine Copula ajustada:")
        logger.info(f"  • Log-verosimilitud: {cop.loglik(data_f64):.4f}")
        logger.info(f"  • AIC:               {cop.aic(data_f64):.4f}")
        logger.info(f"  • BIC:               {cop.bic(data_f64):.4f}")
        return cop

    except Exception as exc:
        raise RuntimeError(f"Error ajustando Vine Copula: {exc}") from exc


# ===========================================================================
# PASO 4 — SIMULACIÓN DE ESCENARIOS SINTÉTICOS
# ===========================================================================

def simulate_stress_scenarios(
    cop: object,
    original_data: pd.DataFrame,
    n_samples: int = N_STRESS_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Genera escenarios de estrés sintéticos muestreando de la Vine Copula
    ajustada y aplicando la Transformación Cuantil Inversa (Quantile Function).

    Proceso:
    ────────
    1. Simular n muestras del modelo: Ũ = {ũ ∈ (0,1)³} ~ C(u₁, u₂, u₃)
    2. Transformación inversa (Inverse PIT):
           x̃ᵢ = F̂ₙ⁻¹(ũᵢ) = np.percentile(xᵢ_original, ũᵢ × 100)
       Esta operación proyecta cada dimensión uniforme de vuelta a su
       dominio físico original usando los percentiles empíricos.

    Los escenarios resultantes preservan:
        - Las distribuciones marginales de cada variable (por construcción).
        - La estructura de dependencia capturada por la cópula.
        - Valores extremos ("colas pesadas") si las cópulas de cola lo capturan.

    Args:
        cop: Objeto Vinecop ajustado.
        original_data: DataFrame con los datos originales (dominio físico).
        n_samples: Número de escenarios sintéticos a generar.
        random_seed: Semilla para reproducibilidad.

    Returns:
        DataFrame con columnas [demand, accel_magnitude, jerk] en dominio real.
    """
    logger.info("─" * 60)
    logger.info(f"PASO 4 — Simulando {n_samples:,} escenarios de estrés")
    logger.info("─" * 60)

    # ── Simulación en el dominio uniforme ──────────────────────────────────
    # API de simulate() ha cambiado entre versiones de pyvinecopulib:
    #   ≥0.6:  cop.simulate(n, seed=int)   ← API actual
    #   <0.6:  cop.simulate(n=int, seeds=[int])
    logger.info("  Muestreando de la Vine Copula ajustada...")
    simulated_u: np.ndarray | None = None
    for _call in [
        lambda: cop.simulate(n_samples, seed=random_seed),
        lambda: cop.simulate(n_samples),
        lambda: cop.simulate(n=n_samples, seeds=[random_seed]),
        lambda: cop.simulate(n_samples=n_samples, seeds=[random_seed]),
    ]:
        try:
            simulated_u = _call()
            break
        except (TypeError, AttributeError):
            continue
    if simulated_u is None:
        raise RuntimeError(
            "cop.simulate() falló con todas las firmas conocidas. "
            "Verifica la versión de pyvinecopulib instalada."
        )

    logger.info(f"  Muestras uniformes shape: {simulated_u.shape}")

    # ── Transformación cuantil inversa ─────────────────────────────────────
    # F̂ₙ⁻¹(u) = percentil(u × 100, datos_originales)
    # Este paso "deshace" la PIT aplicando la función cuantil empírica.
    logger.info("  Aplicando transformación cuantil inversa (Quantile Function)...")

    scenarios: dict[str, np.ndarray] = {}
    for j, col in enumerate(original_data.columns):
        original_values = original_data[col].dropna().values
        u_col = simulated_u[:, j]

        # np.percentile(datos, q) evalúa el percentil q (en %) de los datos
        # Al pasar u × 100 obtenemos F̂⁻¹(u)
        x_synthetic = np.percentile(original_values, u_col * 100.0)
        scenarios[col] = x_synthetic

        logger.info(f"  {col}: sintético rango "
                    f"[{x_synthetic.min():.4f}, {x_synthetic.max():.4f}] "
                    f"(original [{original_values.min():.4f}, {original_values.max():.4f}])")

    scenarios_df = pd.DataFrame(scenarios)

    # ── Añadir columna de percentil de estrés ──────────────────────────────
    # Indicador de "severidad del escenario" basado en la norma L2 de u
    norm_u = np.linalg.norm(simulated_u - 0.5, axis=1)  # distancia del centro
    scenarios_df["stress_percentile"] = rankdata(norm_u) / (len(norm_u) + 1)

    # Escenarios de cola pesada (estrés > percentil 90)
    n_extreme = (scenarios_df["stress_percentile"] > 0.90).sum()
    logger.info(f"\n✓ Escenarios generados: {len(scenarios_df):,} total | "
                f"{n_extreme:,} escenarios extremos (>p90)")

    return scenarios_df


# ===========================================================================
# PASO 5 — EXPORTACIÓN Y RESUMEN
# ===========================================================================

def export_results(
    scenarios_df: pd.DataFrame,
    cop: object,
    pseudo_obs: np.ndarray,
) -> dict:
    """
    Exporta los escenarios a CSV y genera un resumen JSON del modelo
    para documentación de tesis.

    Args:
        scenarios_df: DataFrame con los escenarios sintéticos.
        cop: Objeto Vinecop ajustado.
        pseudo_obs: Pseudo-observaciones usadas para el ajuste.

    Returns:
        Diccionario con el resumen del modelo.
    """
    logger.info("─" * 60)
    logger.info("PASO 5 — Exportando resultados")
    logger.info("─" * 60)

    # ── CSV de escenarios ──────────────────────────────────────────────────
    scenarios_df.to_csv(SCENARIOS_CSV, index=False)
    logger.info(f"✓ stress_scenarios.csv → {SCENARIOS_CSV}")

    # ── Resumen JSON para la tesis ─────────────────────────────────────────
    n, d = pseudo_obs.shape

    summary: dict = {
        "model": "Vine Copula (pyvinecopulib)",
        "variables": VARIABLE_NAMES,
        "n_observations_training": int(n),
        "n_dimensions": int(d),
        "n_scenarios_generated": int(len(scenarios_df)),
        "selection_criterion": "AIC",
        "transform": "Probability Integral Transform (empirical rank / (n+1))",
        "inverse_transform": "Empirical Quantile Function (np.percentile)",
        "random_seed": RANDOM_SEED,
    }

    # Métricas del modelo ajustado
    try:
        data_f64 = pseudo_obs.astype(np.float64)
        summary["log_likelihood"] = float(cop.loglik(data_f64))
        summary["aic"] = float(cop.aic(data_f64))
        summary["bic"] = float(cop.bic(data_f64))
    except Exception as exc:
        logger.warning(f"  No se pudieron extraer métricas del modelo: {exc}")

    # Estadísticas de los escenarios generados
    scenario_stats: dict = {}
    for col in VARIABLE_NAMES:
        if col in scenarios_df.columns:
            s = scenarios_df[col]
            scenario_stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p5": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
    summary["scenario_statistics"] = scenario_stats

    with COPULA_SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info(f"✓ vine_copula_summary.json → {COPULA_SUMMARY_JSON}")

    return summary


# ===========================================================================
# PIPELINE COMPLETO
# ===========================================================================

def run_vine_pipeline(
    n_samples: int = N_STRESS_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> VineGeneratorResult:
    """
    Ejecuta el pipeline completo de generación de escenarios de estrés:

        Datos observados
            → PIT (dominio uniforme)
            → Ajuste Vine Copula
            → Simulación Monte Carlo
            → Transformación Cuantil Inversa
            → Escenarios de estrés en dominio real

    Args:
        n_samples: Número de escenarios sintéticos a generar.
        random_seed: Semilla de aleatoriedad global.

    Returns:
        VineGeneratorResult con todos los artefactos intermedios y finales.
    """
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     VINE COPULA — GENERADOR DE ESCENARIOS DE ESTRÉS      ║")
    logger.info("║     Framework TSC — Módulo 2: Generador Probabilístico   ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── Paso 1: Cargar y construir dataset conjunto ────────────────────────
    joint_data = load_joint_data(random_state=random_seed)

    # ── Paso 2: PIT — Transformación a pseudo-observaciones ───────────────
    pseudo_obs = to_pseudo_observations(joint_data)

    # ── Paso 3: Ajustar la Vine Copula ────────────────────────────────────
    cop = fit_vine_copula(pseudo_obs)

    # ── Paso 4: Simular escenarios de estrés ──────────────────────────────
    scenarios_df = simulate_stress_scenarios(
        cop=cop,
        original_data=joint_data,
        n_samples=n_samples,
        random_seed=random_seed,
    )

    # ── Paso 5: Exportar resultados ───────────────────────────────────────
    summary = export_results(scenarios_df, cop, pseudo_obs)

    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║               PIPELINE COMPLETADO                       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Escenarios guardados en: {SCENARIOS_CSV}")
    logger.info(f"  Log-verosimilitud del modelo: {summary.get('log_likelihood', 'N/A')}")
    logger.info(f"  AIC: {summary.get('aic', 'N/A')} | BIC: {summary.get('bic', 'N/A')}")

    return VineGeneratorResult(
        joint_data=joint_data,
        pseudo_obs=pseudo_obs,
        model=cop,
        scenarios_df=scenarios_df,
        summary=summary,
    )


# ===========================================================================
# PUNTO DE ENTRADA
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Genera escenarios de estrés de tráfico con Vine Copulas."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_STRESS_SAMPLES,
        help=f"Número de escenarios sintéticos a generar (default: {N_STRESS_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Semilla de aleatoriedad (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    result = run_vine_pipeline(n_samples=args.n_samples, random_seed=args.seed)
