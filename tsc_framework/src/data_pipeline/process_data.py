"""
process_data.py
===============
Pipeline de transformación de datos crudos hacia datasets estructurados
listos para el modelado con Vine Copulas.

Módulos:
    - process_macro_flows(): Agrega flujos de vehículos en bins de 15 min.
    - process_micro_behavior(): Calcula aceleración, jerk y estadísticas de telemetría.

Salidas:
    - data/processed/macro_demand.csv
    - data/processed/micro_behavior.csv
    - data/processed/micro_stats.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas base del proyecto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MACRO_RAW_DIR = RAW_DIR / "macro_traffic"
MICRO_RAW_DIR = RAW_DIR / "polidriving"

MACRO_OUTPUT = PROCESSED_DIR / "macro_demand.csv"
MICRO_OUTPUT = PROCESSED_DIR / "micro_behavior.csv"
MICRO_STATS_OUTPUT = PROCESSED_DIR / "micro_stats.json"

# ---------------------------------------------------------------------------
# Constantes de procesamiento
# ---------------------------------------------------------------------------
BIN_SIZE_SECONDS: int = 900          # 15 minutos
OUTLIER_LOW_PERCENTILE: float = 1.0
OUTLIER_HIGH_PERCENTILE: float = 99.0

# Columnas candidatas de aceleración en el dataset de PoliDriving
_ACCEL_COL_CANDIDATES: list[str] = ["acceleration", "accel_x", "accel_y", "accel_z",
                                      "ax", "ay", "az"]
_SPEED_COL_CANDIDATES: list[str] = ["speed", "velocity", "gps_speed"]
_TIME_COL_CANDIDATES: list[str] = ["time", "timestamp", "datetime", "t"]


# ===========================================================================
# UTILIDADES INTERNAS
# ===========================================================================

def _ensure_processed_dir() -> None:
    """Crea el directorio de salida si no existe."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida verificado: {PROCESSED_DIR}")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Busca la primera columna del DataFrame cuyo nombre (case-insensitive)
    coincida con alguno de los candidatos dados.

    Args:
        df: DataFrame a inspeccionar.
        candidates: Lista de nombres de columna candidatos.

    Returns:
        Nombre de la columna encontrada o None si no existe.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_cols:
            return lower_cols[candidate.lower()]
    return None


def _descriptive_stats(series: pd.Series, label: str) -> dict[str, Any]:
    """
    Calcula estadísticas descriptivas completas para una serie numérica.

    Incluye: media, varianza, desviación estándar, asimetría (skewness),
    curtosis (kurtosis) y percentiles clave [1, 5, 25, 50, 75, 95, 99].

    Args:
        series: Serie de pandas con valores numéricos (sin NaN).
        label: Etiqueta descriptiva para el diccionario de resultados.

    Returns:
        Diccionario con las estadísticas calculadas.
    """
    clean = series.dropna()
    if clean.empty:
        logger.warning(f"  Serie '{label}' vacía — estadísticas no calculables.")
        return {}

    percentile_labels = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = np.percentile(clean, percentile_labels).tolist()

    return {
        "label": label,
        "n_samples": int(len(clean)),
        "mean": float(clean.mean()),
        "variance": float(clean.var()),
        "std": float(clean.std()),
        "skewness": float(scipy_stats.skew(clean)),
        "kurtosis": float(scipy_stats.kurtosis(clean)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "percentiles": {
            f"p{p}": v for p, v in zip(percentile_labels, percentile_values)
        },
    }


# ===========================================================================
# MÓDULO 1 — FLUJOS MACROSCÓPICOS (Hangzhou / Jinan / CoLight)
# ===========================================================================

def _parse_flow_file(filepath: Path) -> pd.Series | None:
    """
    Lee un archivo de flujo CityFlow (JSON o texto) y extrae los
    tiempos de inicio de cada vehículo como una Serie de enteros.

    El formato esperado es una lista de objetos con clave ``startTime``
    (en segundos desde el inicio de la simulación).

    Args:
        filepath: Ruta al archivo ``flow.json`` o ``flow.txt``.

    Returns:
        Series de int con los startTime, o None si el archivo es inválido.
    """
    try:
        with filepath.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # El archivo puede ser una lista directamente o un dict con clave 'flow'
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            # Buscar la clave que contiene la lista de vehículos
            for key in ("flow", "vehicles", "trips", "data"):
                if key in raw and isinstance(raw[key], list):
                    records = raw[key]
                    break
            else:
                # Último recurso: tomar el primer valor que sea lista
                lists = [v for v in raw.values() if isinstance(v, list)]
                if not lists:
                    logger.warning(f"  [{filepath.name}] Estructura JSON no reconocida.")
                    return None
                records = lists[0]
        else:
            logger.warning(f"  [{filepath.name}] Tipo JSON inesperado: {type(raw)}")
            return None

        # Extraer startTime de cada registro
        start_times: list[float] = []
        for rec in records:
            # Soporta {'startTime': N} y {'vehicle': {...}, 'startTime': N}
            if isinstance(rec, dict):
                st = rec.get("startTime") or rec.get("start_time") or rec.get("depart")
                if st is not None:
                    try:
                        start_times.append(float(st))
                    except (TypeError, ValueError):
                        pass

        if not start_times:
            logger.warning(f"  [{filepath.name}] No se encontraron startTime válidos.")
            return None

        logger.info(f"  [{filepath.name}] {len(start_times):,} vehículos leídos.")
        return pd.Series(start_times, dtype="float64")

    except json.JSONDecodeError as exc:
        logger.error(f"  [{filepath.name}] JSON malformado — {exc}")
        return None
    except OSError as exc:
        logger.error(f"  [{filepath.name}] Error de lectura — {exc}")
        return None
    except Exception as exc:
        logger.error(f"  [{filepath.name}] Error inesperado — {exc}")
        return None


def process_macro_flows() -> pd.DataFrame | None:
    """
    Procesa los archivos de flujo CityFlow/CoLight para generar una serie
    temporal de demanda de tráfico agregada por intervalos de 15 minutos.

    Busca recursivamente archivos ``flow.json`` y ``flow.txt`` en
    ``data/raw/macro_traffic/``, extrae los ``startTime`` de cada vehículo,
    los agrega en bins temporales de 900 s y exporta el resultado a
    ``data/processed/macro_demand.csv``.

    Returns:
        DataFrame con columnas [time_bin_s, vehicle_count, source_file],
        o None si no se procesó ningún archivo.
    """
    logger.info("=" * 60)
    logger.info("MACRO — Iniciando procesamiento de flujos de tráfico")
    logger.info("=" * 60)

    if not MACRO_RAW_DIR.exists():
        logger.error(f"Directorio macro no encontrado: {MACRO_RAW_DIR}")
        return None

    # Búsqueda recursiva de archivos de flujo
    flow_files = (
        list(MACRO_RAW_DIR.rglob("flow.json"))
        + list(MACRO_RAW_DIR.rglob("flow.txt"))
        # También captura archivos con 'flow' en el nombre, ej. anon_4_4_hangzhou_real.json
        + [
            f for f in MACRO_RAW_DIR.rglob("*.json")
            if "flow" in f.name.lower() or "anon" in f.name.lower()
        ]
    )
    # Deduplicar manteniendo orden
    seen: set[Path] = set()
    unique_flow_files: list[Path] = []
    for fp in flow_files:
        if fp not in seen:
            seen.add(fp)
            unique_flow_files.append(fp)

    if not unique_flow_files:
        logger.warning("No se encontraron archivos de flujo. Verifica data/raw/macro_traffic/")
        return None

    logger.info(f"Archivos de flujo encontrados: {len(unique_flow_files)}")

    all_rows: list[dict[str, Any]] = []

    for flow_file in unique_flow_files:
        logger.info(f"Procesando: {flow_file.relative_to(PROJECT_ROOT)}")
        start_times = _parse_flow_file(flow_file)
        if start_times is None or start_times.empty:
            continue

        # Asignar bin temporal (floor division → bin en segundos)
        bins = (start_times // BIN_SIZE_SECONDS) * BIN_SIZE_SECONDS
        counts = bins.value_counts().sort_index().reset_index()
        counts.columns = ["time_bin_s", "vehicle_count"]
        counts["time_bin_min"] = counts["time_bin_s"] / 60.0
        counts["source_file"] = flow_file.name
        counts["source_city"] = flow_file.parts[
            # Inferir ciudad desde la ruta relativa
            min(len(flow_file.parts) - 1, flow_file.parts.index(
                next((p for p in flow_file.parts if p in MACRO_RAW_DIR.parts), flow_file.parts[-2])
            ) + 1)
        ] if any(p in MACRO_RAW_DIR.parts for p in flow_file.parts) else "unknown"

        # Forma más robusta de inferir ciudad
        rel_parts = flow_file.relative_to(MACRO_RAW_DIR).parts
        counts["source_city"] = rel_parts[0] if rel_parts else "unknown"

        all_rows.append(counts)

    if not all_rows:
        logger.error("No se pudo procesar ningún archivo de flujo.")
        return None

    demand_df = pd.concat(all_rows, ignore_index=True)
    demand_df = demand_df.sort_values(["source_city", "source_file", "time_bin_s"])

    demand_df.to_csv(MACRO_OUTPUT, index=False)
    logger.info(f"✓ macro_demand.csv exportado → {MACRO_OUTPUT}")
    logger.info(f"  Filas totales: {len(demand_df):,} | "
                f"Ciudades: {demand_df['source_city'].nunique()} | "
                f"Archivos: {demand_df['source_file'].nunique()}")

    return demand_df


# ===========================================================================
# MÓDULO 2 — COMPORTAMIENTO MICROSCÓPICO (PoliDriving)
# ===========================================================================

def _load_polidriving_csv(filepath: Path) -> pd.DataFrame | None:
    """
    Carga un archivo CSV de telemetría PoliDriving con manejo robusto
    de encodings y separadores.

    Args:
        filepath: Ruta al archivo CSV de telemetría.

    Returns:
        DataFrame cargado o None si el archivo es inválido o vacío.
    """
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            if df.empty:
                logger.warning(f"  [{filepath.name}] Archivo vacío.")
                return None
            logger.info(f"  [{filepath.name}] {len(df):,} filas, {len(df.columns)} columnas "
                        f"(encoding={encoding})")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            logger.warning(f"  [{filepath.name}] Sin datos.")
            return None
        except pd.errors.ParserError as exc:
            logger.error(f"  [{filepath.name}] Error de parseo CSV — {exc}")
            return None
        except Exception as exc:
            logger.error(f"  [{filepath.name}] Error inesperado — {exc}")
            return None

    logger.error(f"  [{filepath.name}] No se pudo decodificar el archivo.")
    return None


def _compute_accel_magnitude(df: pd.DataFrame) -> pd.Series | None:
    """
    Calcula la magnitud de aceleración a partir de los ejes disponibles
    o la deriva de la velocidad GPS.

    Estrategia:
        1. Si existen columnas de ejes (X, Y, Z) → magnitud euclidiana.
        2. Si existe columna ``acceleration`` escalar → valor absoluto.
        3. Si existe columna ``speed`` → derivada numérica (Δv/Δt).

    Args:
        df: DataFrame con columnas de telemetría.

    Returns:
        Serie con la magnitud de aceleración (m/s²), o None si no es posible.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Estrategia 1 — Ejes del acelerómetro
    axes_found = []
    for axis in ("ax", "ay", "az", "accel_x", "accel_y", "accel_z"):
        if axis in cols_lower:
            axes_found.append(cols_lower[axis])

    if len(axes_found) >= 2:
        acc_arrays = [pd.to_numeric(df[col], errors="coerce") for col in axes_found]
        magnitude = np.sqrt(sum(a**2 for a in acc_arrays))
        logger.debug("  Magnitud calculada desde ejes del acelerómetro.")
        return magnitude

    # Estrategia 2 — Columna de aceleración escalar
    accel_col = _find_column(df, _ACCEL_COL_CANDIDATES)
    if accel_col is not None:
        series = pd.to_numeric(df[accel_col], errors="coerce")
        if series.notna().sum() > 0:
            logger.debug(f"  Usando columna de aceleración escalar: '{accel_col}'")
            return series.abs()

    # Estrategia 3 — Derivada de velocidad GPS
    speed_col = _find_column(df, _SPEED_COL_CANDIDATES)
    time_col = _find_column(df, _TIME_COL_CANDIDATES)

    if speed_col is not None:
        speed = pd.to_numeric(df[speed_col], errors="coerce")
        if time_col is not None:
            # Intentar parsear tiempo para calcular Δt real
            try:
                t = pd.to_datetime(df[time_col], format="%H:%M:%S", errors="coerce")
                delta_t = t.diff().dt.total_seconds().fillna(1.0).clip(lower=0.1)
                derived = speed.diff().abs() / delta_t
                logger.debug("  Magnitud derivada de GPS speed con Δt real.")
                return derived
            except Exception:
                pass
        # Sin tiempo válido → asumir 1 Hz
        derived = speed.diff().abs()
        logger.debug("  Magnitud derivada de GPS speed (Δt=1s asumido).")
        return derived

    return None


def _compute_jerk(accel: pd.Series, time_col: pd.Series | None = None) -> pd.Series:
    """
    Calcula el jerk (derivada de la aceleración respecto al tiempo) en m/s³.

    Si no se dispone de una columna de tiempo, se asume muestreo uniforme
    a 1 Hz (Δt = 1 segundo).

    Args:
        accel: Serie de aceleración en m/s².
        time_col: Serie de tiempos opcionales (string HH:MM:SS o datetime).

    Returns:
        Serie con el jerk en m/s³.
    """
    delta_accel = accel.diff()

    if time_col is not None:
        try:
            t = pd.to_datetime(time_col, format="%H:%M:%S", errors="coerce")
            delta_t = t.diff().dt.total_seconds().fillna(1.0).clip(lower=0.1)
            return (delta_accel / delta_t).fillna(0.0)
        except Exception:
            pass

    # Δt = 1 s asumido
    return delta_accel.fillna(0.0)


def _apply_outlier_filter(
    df: pd.DataFrame,
    columns: list[str],
    low: float = OUTLIER_LOW_PERCENTILE,
    high: float = OUTLIER_HIGH_PERCENTILE,
) -> pd.DataFrame:
    """
    Filtra filas donde cualquiera de las columnas especificadas esté fuera
    del rango [percentil_low, percentil_high].

    Args:
        df: DataFrame de entrada.
        columns: Lista de columnas sobre las que aplicar el filtro.
        low: Percentil inferior (default 1).
        high: Percentil superior (default 99).

    Returns:
        DataFrame filtrado.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        p_low = series.quantile(low / 100)
        p_high = series.quantile(high / 100)
        mask &= series.between(p_low, p_high)

    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info(f"  Outliers eliminados: {n_removed:,} filas "
                    f"({n_removed / len(df) * 100:.1f}%)")
    return df[mask].copy()


def process_micro_behavior() -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """
    Procesa los archivos CSV de telemetría PoliDriving para extraer la
    "firma conductual" de cada conductor.

    Flujo de procesamiento:
        1. Búsqueda recursiva de CSVs en ``data/raw/polidriving/``.
        2. Cálculo de magnitud de aceleración y jerk por viaje.
        3. Filtrado de outliers en percentiles [1, 99].
        4. Concatenación de todos los viajes.
        5. Cálculo de estadísticas descriptivas globales.
        6. Exportación a CSV y JSON.

    Returns:
        Tupla (DataFrame limpio, dict de estadísticas) o (None, None) en error.
    """
    logger.info("=" * 60)
    logger.info("MICRO — Iniciando procesamiento de telemetría PoliDriving")
    logger.info("=" * 60)

    if not MICRO_RAW_DIR.exists():
        logger.error(f"Directorio PoliDriving no encontrado: {MICRO_RAW_DIR}")
        return None, None

    # Búsqueda recursiva de CSVs (excluir archivos de actividad y metadatos)
    all_csvs = [
        f for f in MICRO_RAW_DIR.rglob("*.csv")
        if "ACTIVITY" not in f.name.upper()
        and "README" not in f.name.upper()
    ]

    if not all_csvs:
        logger.warning("No se encontraron archivos CSV en data/raw/polidriving/")
        return None, None

    logger.info(f"Archivos CSV encontrados: {len(all_csvs)}")

    trip_dfs: list[pd.DataFrame] = []

    for csv_file in sorted(all_csvs):
        logger.info(f"Procesando: {csv_file.relative_to(PROJECT_ROOT)}")

        df = _load_polidriving_csv(csv_file)
        if df is None:
            continue

        # --- Metadata del viaje ---
        rel_parts = csv_file.relative_to(MICRO_RAW_DIR).parts
        driver_id = rel_parts[0] if len(rel_parts) > 0 else "unknown"
        trip_id = rel_parts[1] if len(rel_parts) > 1 else csv_file.stem

        # --- Columna de tiempo ---
        time_col_name = _find_column(df, _TIME_COL_CANDIDATES)

        # --- Calcular magnitud de aceleración ---
        accel_mag = _compute_accel_magnitude(df)
        if accel_mag is None:
            logger.warning(f"  No se pudo calcular aceleración para {csv_file.name}. "
                           f"Columnas disponibles: {list(df.columns[:10])}")
            continue

        df["accel_magnitude"] = accel_mag

        # --- Calcular jerk ---
        time_series = df[time_col_name] if time_col_name else None
        df["jerk"] = _compute_jerk(df["accel_magnitude"], time_series)

        # --- Seleccionar columnas de salida disponibles ---
        output_cols = ["accel_magnitude", "jerk"]

        # Añadir columnas de contexto si existen
        for optional in ["speed", "latitude", "longitude", "altitude",
                         "heart_rate", "acceleration"]:
            found = _find_column(df, [optional])
            if found:
                output_cols.append(found)

        trip_subset = df[output_cols].copy()

        # --- Convertir a numérico y eliminar NaN estructurales ---
        for col in ["accel_magnitude", "jerk"]:
            trip_subset[col] = pd.to_numeric(trip_subset[col], errors="coerce")

        trip_subset = trip_subset.dropna(subset=["accel_magnitude", "jerk"])

        if trip_subset.empty:
            logger.warning(f"  Viaje {csv_file.name} queda vacío tras limpieza de NaN.")
            continue

        # --- Filtrado de outliers ---
        trip_subset = _apply_outlier_filter(
            trip_subset,
            columns=["accel_magnitude", "jerk"],
        )

        if trip_subset.empty:
            logger.warning(f"  Viaje {csv_file.name} queda vacío tras filtrado de outliers.")
            continue

        # --- Metadatos de trazabilidad ---
        trip_subset.insert(0, "driver_id", driver_id)
        trip_subset.insert(1, "trip_id", trip_id)
        trip_subset.insert(2, "source_file", csv_file.name)

        trip_dfs.append(trip_subset)
        logger.info(f"  ✓ Viaje procesado: {len(trip_subset):,} muestras válidas.")

    if not trip_dfs:
        logger.error("No se procesó ningún viaje de PoliDriving.")
        return None, None

    # --- Concatenar todos los viajes ---
    behavior_df = pd.concat(trip_dfs, ignore_index=True)
    logger.info(f"\nTotal de muestras concatenadas: {len(behavior_df):,} | "
                f"Conductores: {behavior_df['driver_id'].nunique()} | "
                f"Viajes: {behavior_df['trip_id'].nunique()}")

    # --- Calcular estadísticas descriptivas globales ---
    stats: dict[str, Any] = {
        "dataset": "PoliDriving — Telemetría Microscópica Ecuador",
        "total_samples": int(len(behavior_df)),
        "n_drivers": int(behavior_df["driver_id"].nunique()),
        "n_trips": int(behavior_df["trip_id"].nunique()),
        "drivers": behavior_df["driver_id"].unique().tolist(),
        "accel_magnitude": _descriptive_stats(
            behavior_df["accel_magnitude"], "Magnitud de Aceleración (m/s²)"
        ),
        "jerk": _descriptive_stats(
            behavior_df["jerk"], "Jerk (m/s³)"
        ),
    }

    # Estadísticas por conductor
    per_driver: dict[str, Any] = {}
    for driver, group in behavior_df.groupby("driver_id"):
        per_driver[str(driver)] = {
            "n_samples": int(len(group)),
            "n_trips": int(group["trip_id"].nunique()),
            "accel_magnitude": _descriptive_stats(
                group["accel_magnitude"], f"Accel [{driver}]"
            ),
            "jerk": _descriptive_stats(
                group["jerk"], f"Jerk [{driver}]"
            ),
        }
    stats["per_driver"] = per_driver

    # --- Exportar CSV ---
    behavior_df.to_csv(MICRO_OUTPUT, index=False)
    logger.info(f"✓ micro_behavior.csv exportado → {MICRO_OUTPUT}")

    # --- Exportar JSON de estadísticas ---
    with MICRO_STATS_OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info(f"✓ micro_stats.json exportado → {MICRO_STATS_OUTPUT}")

    return behavior_df, stats


# ===========================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ===========================================================================

def main() -> None:
    """
    Ejecuta el pipeline completo de transformación de datos:
        1. Procesamiento macroscópico (flujos de tráfico).
        2. Procesamiento microscópico (telemetría de conductores).
    """
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║        PIPELINE DE TRANSFORMACIÓN DE DATOS               ║")
    logger.info("║        Framework TSC — Módulo 1: Data Pipeline           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    _ensure_processed_dir()

    # ── Módulo Macro ──────────────────────────────────────────────────────
    macro_df = process_macro_flows()
    if macro_df is not None:
        logger.info(f"\n[MACRO] Resumen:\n{macro_df.groupby('source_city')['vehicle_count'].sum()}\n")
    else:
        logger.warning("[MACRO] No se generaron datos de demanda.")

    # ── Módulo Micro ──────────────────────────────────────────────────────
    micro_df, micro_stats = process_micro_behavior()
    if micro_stats is not None:
        logger.info(
            f"\n[MICRO] Aceleración — Media: "
            f"{micro_stats['accel_magnitude'].get('mean', 'N/A'):.4f} m/s² | "
            f"Jerk — Media: {micro_stats['jerk'].get('mean', 'N/A'):.4f} m/s³"
        )
    else:
        logger.warning("[MICRO] No se generaron datos de comportamiento.")

    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║               PIPELINE COMPLETADO                       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Archivos generados en: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
