"""
route_generator.py
==================
Módulo 2→3 Bridge: Convierte escenarios de estrés probabilísticos (Vine Copula)
en archivos de rutas SUMO (.rou.xml) usando topología real de Hangzhou/CoLight.

Mapeo matemático (Modelo de Seguimiento Krauß):
    accel_magnitude  → accel  ∈ [1.0, 3.5]  m/s²
    accel_magnitude  → decel  ∈ [2.0, 5.0]  m/s²   (decel = accel × 1.6, clipped)
    |jerk|           → sigma  ∈ [0.1, 0.9]  (imperfección del conductor)
    |jerk|           → tau    ∈ [1.0, 2.5]  s       (tiempo de reacción)
    demand           → escala del headway / probabilidad de inclusión de vehículo
"""

from __future__ import annotations

import json
import logging
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
MACRO_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "macro_traffic"
OUTPUT_DIR = PROJECT_ROOT / "sumo_configs" / "routes"

SCENARIOS_CSV = PROCESSED_DIR / "stress_scenarios.csv"

# ---------------------------------------------------------------------------
# Constantes del modelo Krauß
# ---------------------------------------------------------------------------
ACCEL_MIN, ACCEL_MAX = 1.0, 3.5          # m/s²
DECEL_MIN, DECEL_MAX = 2.0, 5.0          # m/s²
DECEL_ACCEL_RATIO = 1.6                  # decel ≈ 1.6 × accel (empirical)
SIGMA_MIN, SIGMA_MAX = 0.1, 0.9          # dimensionless imperfection
TAU_MIN, TAU_MAX = 1.0, 2.5             # seconds reaction time
MIN_GAP = 1.5                            # m
MAX_SPEED = 15.0                         # m/s (~54 km/h, urban)

# Rango de referencia del jerk absoluto para normalización
# (calculado del pipeline: p1≈0, p99≈~4 m/s³ típico)
JERK_REF_MAX = 4.0                       # m/s³  — umbral de saturación

# Fases de prueba
N_SCENARIOS_PHASE1 = 100


# ===========================================================================
# Dataclass de parámetros Krauß
# ===========================================================================

@dataclass
class KraussParams:
    """Parámetros del modelo de seguimiento vehicular de Krauß para SUMO."""
    accel: float
    decel: float
    sigma: float
    tau: float
    min_gap: float = MIN_GAP
    max_speed: float = MAX_SPEED

    def to_attribs(self) -> dict[str, str]:
        """Serializa a diccionario de atributos XML (todos string)."""
        return {
            "accel":    f"{self.accel:.4f}",
            "decel":    f"{self.decel:.4f}",
            "sigma":    f"{self.sigma:.4f}",
            "tau":      f"{self.tau:.4f}",
            "minGap":   f"{self.min_gap:.2f}",
            "maxSpeed": f"{self.max_speed:.3f}",
        }


# ===========================================================================
# PASO 1 — Carga de escenarios y datos de flujo reales
# ===========================================================================

def load_stress_scenarios(
    path: Path = SCENARIOS_CSV,
    n_scenarios: int = N_SCENARIOS_PHASE1,
) -> pd.DataFrame:
    """
    Carga los primeros ``n_scenarios`` escenarios del CSV de estrés generado
    por la Vine Copula.

    Args:
        path: Ruta a ``stress_scenarios.csv``.
        n_scenarios: Número de escenarios a cargar (fase de prueba = 100).

    Returns:
        DataFrame con columnas [demand, accel_magnitude, jerk, stress_percentile].

    Raises:
        FileNotFoundError: Si el CSV no existe (ejecutar vine_generator.py primero).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró: {path}\n"
            "Ejecuta primero: python src/copulas/vine_generator.py"
        )

    df = pd.read_csv(path, nrows=n_scenarios)
    required = ["demand", "accel_magnitude", "jerk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {path.name}: {missing}")

    # Asegurar tipos numéricos
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    logger.info(f"✓ Escenarios cargados: {len(df)} filas desde {path.name}")
    return df


def find_flow_file(
    city_dir: Path = MACRO_RAW_DIR / "Hangzhou",
    preferred_name: str = "anon_4_4_hangzhou_real.json",
) -> Path:
    """
    Busca recursivamente un archivo de flujo CityFlow dentro de ``city_dir``.

    La búsqueda prioriza el nombre ``preferred_name``; si no se encuentra,
    toma el primer ``.json`` cuyo nombre contenga "flow" o "anon".

    Args:
        city_dir: Directorio raíz de la ciudad (ej. macro_traffic/Hangzhou).
        preferred_name: Nombre de archivo preferido.

    Returns:
        Path al archivo de flujo encontrado.

    Raises:
        FileNotFoundError: Si no hay ningún JSON de flujo en el directorio.
    """
    if not city_dir.exists():
        raise FileNotFoundError(f"Directorio de ciudad no encontrado: {city_dir}")

    # Búsqueda 1: nombre exacto preferido
    for candidate in city_dir.rglob(preferred_name):
        logger.info(f"  Archivo de flujo preferido encontrado: {candidate.relative_to(PROJECT_ROOT)}")
        return candidate

    # Búsqueda 2: cualquier JSON que parezca flujo
    fallback_patterns = ["flow.json", "flow.txt"]
    for pattern in fallback_patterns:
        matches = list(city_dir.rglob(pattern))
        if matches:
            logger.info(f"  Usando flujo alternativo: {matches[0].relative_to(PROJECT_ROOT)}")
            return matches[0]

    # Búsqueda 3: cualquier JSON con 'anon' o 'flow' en el nombre
    for fp in city_dir.rglob("*.json"):
        if any(kw in fp.name.lower() for kw in ("flow", "anon", "trip")):
            logger.info(f"  Usando flujo (búsqueda amplia): {fp.relative_to(PROJECT_ROOT)}")
            return fp

    raise FileNotFoundError(
        f"No se encontró ningún archivo de flujo en {city_dir}.\n"
        f"Verifica que los datos macro estén descargados."
    )


def load_flow_records(flow_path: Path) -> list[dict[str, Any]]:
    """
    Lee y parsea un archivo de flujo CityFlow/CoLight.

    Estructura esperada (lista de objetos):
        [{"vehicle": {...}, "route": ["edge_A", "edge_B"], "startTime": 0, ...}, ...]

    Soporta también el formato dict con clave "flow" o "trips" en la raíz.

    Args:
        flow_path: Ruta al archivo JSON de flujo.

    Returns:
        Lista de diccionarios de vehículos/flujos.

    Raises:
        ValueError: Si la estructura del JSON no es reconocida.
    """
    with flow_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        for key in ("flow", "vehicles", "trips", "data"):
            if key in raw and isinstance(raw[key], list):
                records = raw[key]
                break
        else:
            lists = [v for v in raw.values() if isinstance(v, list)]
            if not lists:
                raise ValueError(f"Estructura JSON no reconocida en {flow_path.name}")
            records = lists[0]
    else:
        raise ValueError(f"Tipo JSON inesperado en {flow_path.name}: {type(raw)}")

    # Filtrar registros que tengan al menos 'route' y 'startTime'
    valid = [
        r for r in records
        if isinstance(r, dict)
        and isinstance(r.get("route"), list)
        and len(r["route"]) >= 1
        and r.get("startTime") is not None
    ]

    logger.info(f"  Registros de flujo válidos: {len(valid):,} / {len(records):,}")
    return valid


# ===========================================================================
# PASO 2 — Mapeo matemático al modelo Krauß
# ===========================================================================

def _normalize(value: float, v_min: float, v_max: float, eps: float = 1e-9) -> float:
    """Normaliza ``value`` al rango [0, 1] dado [v_min, v_max]."""
    return float(np.clip((value - v_min) / (v_max - v_min + eps), 0.0, 1.0))


def _linear_map(
    norm_value: float, out_min: float, out_max: float
) -> float:
    """Mapea un valor normalizado [0,1] al rango de salida [out_min, out_max]."""
    return out_min + norm_value * (out_max - out_min)


def map_to_krauss(
    accel_magnitude: float,
    jerk: float,
    accel_ref_min: float = 0.0,
    accel_ref_max: float = 3.5,
    jerk_ref_max: float = JERK_REF_MAX,
) -> KraussParams:
    """
    Transforma los parámetros comportamentales del CSV en parámetros del
    modelo de seguimiento de Krauß para SUMO.

    Mapeos:
        accel  = lineal(accel_magnitude, [accel_ref_min, accel_ref_max], [ACCEL_MIN, ACCEL_MAX])
        decel  = clip(accel × DECEL_ACCEL_RATIO, DECEL_MIN, DECEL_MAX)
        sigma  = lineal(|jerk|, [0, jerk_ref_max], [SIGMA_MIN, SIGMA_MAX])
                 → jerk alto ≡ conducción errática → sigma cercano a 0.9
        tau    = lineal(|jerk|, [0, jerk_ref_max], [TAU_MIN, TAU_MAX])
                 → jerk alto ≡ mayor tiempo de reacción → tau hasta 2.5 s

    Args:
        accel_magnitude: Magnitud de aceleración del escenario (m/s²).
        jerk: Derivada de la aceleración del escenario (m/s³).
        accel_ref_min: Mínimo del rango de referencia de accel_magnitude.
        accel_ref_max: Máximo del rango de referencia de accel_magnitude.
        jerk_ref_max: Valor de jerk absoluto que satura sigma/tau a su máximo.

    Returns:
        KraussParams con todos los parámetros del modelo.
    """
    # Normalización de accel_magnitude → [0, 1]
    norm_accel = _normalize(accel_magnitude, accel_ref_min, accel_ref_max)

    # accel: a mayor accel_magnitude observada → mayor parámetro SUMO
    sumo_accel = _linear_map(norm_accel, ACCEL_MIN, ACCEL_MAX)

    # decel: proporcionalmente mayor a accel, limitada al rango seguro
    sumo_decel = float(np.clip(sumo_accel * DECEL_ACCEL_RATIO, DECEL_MIN, DECEL_MAX))

    # Normalización de |jerk| → [0, 1] (saturación en jerk_ref_max)
    norm_jerk = _normalize(abs(jerk), 0.0, jerk_ref_max)

    # sigma: jerk alto → conducción errática → sigma alto [0.1, 0.9]
    sumo_sigma = _linear_map(norm_jerk, SIGMA_MIN, SIGMA_MAX)

    # tau: jerk alto → menor atención → tau alto [1.0, 2.5]
    sumo_tau = _linear_map(norm_jerk, TAU_MIN, TAU_MAX)

    return KraussParams(
        accel=round(sumo_accel, 4),
        decel=round(sumo_decel, 4),
        sigma=round(sumo_sigma, 4),
        tau=round(sumo_tau, 4),
    )


# ===========================================================================
# PASO 3 — Generador de XML topológico
# ===========================================================================

def _compute_demand_factor(demand: float, demand_series: pd.Series) -> float:
    """
    Calcula el factor de densidad vehicular [0.2, 1.0] a partir del valor
    de demanda del escenario y el rango observado.

    Un factor < 1.0 reduce la densidad de vehículos incluidos: solo se
    incluye un vehículo si ``random() < demand_factor``.

    Args:
        demand: Valor de demanda del escenario (veh/intervalo).
        demand_series: Serie de demandas para calcular el rango de referencia.

    Returns:
        Factor de densidad en [0.2, 1.0].
    """
    d_min = demand_series.min()
    d_max = demand_series.max()
    norm = _normalize(demand, d_min, d_max)
    # Rango [0.2, 1.0]: incluso el escenario de menor demanda incluye 20% de vehículos
    return round(_linear_map(norm, 0.2, 1.0), 4)


def generate_route_xml(
    scenario_idx: int,
    krauss: KraussParams,
    flow_records: list[dict[str, Any]],
    demand_factor: float,
    city_tag: str = "hangzhou",
    rng: np.random.Generator | None = None,
) -> ET.Element:
    """
    Genera el árbol XML de un archivo ``.rou.xml`` de SUMO para un escenario.

    Estructura generada:
        <routes>
            <vType id="latam_car" accel="..." decel="..." sigma="..." tau="..." .../>
            <vehicle id="veh_0" type="latam_car" depart="0.0" ...>
                <route edges="road_A road_B road_C"/>
            </vehicle>
            ...
        </routes>

    La etiqueta ``<vehicle>`` se usa cuando startTime == endTime (un solo vehículo).
    La etiqueta ``<flow>`` se usa cuando hay un intervalo de salida definido.

    Args:
        scenario_idx: Índice del escenario (para IDs únicos).
        krauss: Parámetros del modelo Krauß para este escenario.
        flow_records: Lista de registros del flow.json de la ciudad real.
        demand_factor: Fracción de vehículos a incluir [0.2, 1.0].
        city_tag: Etiqueta de ciudad para IDs de vehículos.
        rng: Generador de números aleatorios para el submuestreo por demanda.

    Returns:
        Elemento raíz ``<routes>`` de ElementTree.
    """
    if rng is None:
        rng = np.random.default_rng(scenario_idx)

    # ── Raíz y cabecera ────────────────────────────────────────────────────
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set(
        "xsi:noNamespaceSchemaLocation",
        "http://sumo.dlr.de/xsd/routes_file.xsd",
    )

    # ── Definición del tipo de vehículo ────────────────────────────────────
    vtype_attribs = {
        "id":     "latam_car",
        "length": "5.00",
        "width":  "2.00",
        **krauss.to_attribs(),
        "departSpeed": "0",
        "departLane":  "best",
    }
    ET.SubElement(root, "vType", attrib=vtype_attribs)

    # ── Generación de vehículos/flujos con rutas reales ────────────────────
    veh_counter = 0
    for rec_idx, record in enumerate(flow_records):
        # Aplicar factor de demanda: submuestreo aleatorio
        if rng.random() > demand_factor:
            continue

        route_edges: list[str] = record.get("route", [])
        if not route_edges:
            continue

        start_time = float(record.get("startTime", 0))
        end_time = float(record.get("endTime", start_time))
        interval = float(record.get("interval", 1.0))
        edges_str = " ".join(route_edges)

        veh_id = f"s{scenario_idx:03d}_{city_tag}_{veh_counter:04d}"

        if start_time == end_time:
            # Vehículo individual → <vehicle> con <route> inline
            veh_elem = ET.SubElement(root, "vehicle", attrib={
                "id":     veh_id,
                "type":   "latam_car",
                "depart": f"{start_time:.1f}",
            })
            ET.SubElement(veh_elem, "route", attrib={"edges": edges_str})
        else:
            # Flujo con intervalo → <flow>
            # period = segundos entre salidas de vehículos
            period = max(interval, 1.0)
            ET.SubElement(root, "flow", attrib={
                "id":     veh_id,
                "type":   "latam_car",
                "begin":  f"{start_time:.1f}",
                "end":    f"{end_time:.1f}",
                "period": f"{period:.1f}",
                "route":  edges_str,   # edges directamente en <flow>
            })

        veh_counter += 1

    logger.debug(
        f"  Escenario {scenario_idx:03d}: "
        f"{veh_counter} vehículos incluidos "
        f"(factor demanda={demand_factor:.2f}, total flujo={len(flow_records)})"
    )
    return root


# ===========================================================================
# PASO 4 — Pretty-print y exportación
# ===========================================================================

def _pretty_xml(root: ET.Element) -> str:
    """
    Formatea un árbol ElementTree como XML indentado legible (pretty-print).

    Usa ``xml.dom.minidom`` para añadir saltos de línea e indentación de 4
    espacios, y elimina la línea en blanco que minidom inserta a veces.

    Args:
        root: Elemento raíz del árbol XML.

    Returns:
        String XML con declaración y formato indentado.
    """
    raw_str = ET.tostring(root, encoding="unicode", xml_declaration=False)
    dom = minidom.parseString(raw_str)
    pretty = dom.toprettyxml(indent="    ", encoding=None)
    # Eliminar la línea de declaración <?xml?> duplicada que agrega minidom
    lines = pretty.split("\n")
    # Reemplazar por declaración UTF-8 estándar
    lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    return "\n".join(line for line in lines if line.strip())


def export_scenario(
    xml_root: ET.Element,
    scenario_idx: int,
    city_tag: str = "hangzhou",
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Guarda el XML de un escenario en disco con nombre estandarizado.

    Args:
        xml_root: Elemento raíz ``<routes>`` generado.
        scenario_idx: Índice del escenario (usado en el nombre del archivo).
        city_tag: Etiqueta de ciudad para el nombre del archivo.
        output_dir: Directorio de salida.

    Returns:
        Path del archivo ``.rou.xml`` creado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"scenario_{scenario_idx:03d}_{city_tag}.rou.xml"
    out_path = output_dir / filename

    pretty_content = _pretty_xml(xml_root)
    out_path.write_text(pretty_content, encoding="utf-8")
    return out_path


# ===========================================================================
# PIPELINE COMPLETO
# ===========================================================================

def run_route_generation(
    n_scenarios: int = N_SCENARIOS_PHASE1,
    city: str = "Hangzhou",
    flow_filename: str = "anon_4_4_hangzhou_real.json",
    output_dir: Path = OUTPUT_DIR,
    random_seed: int = 42,
) -> list[Path]:
    """
    Ejecuta el pipeline completo de generación de rutas SUMO:

        stress_scenarios.csv
            → Mapeo Krauß por escenario
            → XML topológico con rutas reales de Hangzhou
            → scenario_NNN_hangzhou.rou.xml × N_SCENARIOS

    Args:
        n_scenarios: Número de escenarios a procesar.
        city: Nombre de la carpeta de ciudad en macro_traffic/.
        flow_filename: Nombre del archivo de flujo preferido.
        output_dir: Directorio de salida para los .rou.xml.
        random_seed: Semilla de aleatoriedad para el submuestreo de demanda.

    Returns:
        Lista de Paths de los archivos .rou.xml generados.
    """
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   GENERADOR DE RUTAS SUMO — Escenarios Probabilísticos   ║")
    logger.info("║   Framework TSC — Módulo 2→3 Bridge                     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    city_tag = city.lower()
    rng = np.random.default_rng(random_seed)

    # ── Paso 1a: Cargar escenarios de estrés ──────────────────────────────
    logger.info("─" * 60)
    logger.info("PASO 1 — Cargando datos de entrada")
    logger.info("─" * 60)
    scenarios_df = load_stress_scenarios(n_scenarios=n_scenarios)

    # ── Paso 1b: Cargar flujo real de la ciudad ───────────────────────────
    city_dir = MACRO_RAW_DIR / city
    flow_path = find_flow_file(city_dir=city_dir, preferred_name=flow_filename)
    flow_records = load_flow_records(flow_path)

    if not flow_records:
        raise ValueError(f"No hay registros válidos en {flow_path.name}")

    logger.info(f"  Ciudad: {city} | Rutas reales: {len(flow_records):,}")

    # Estadísticas de referencia para normalización
    accel_ref_min = float(scenarios_df["accel_magnitude"].quantile(0.01))
    accel_ref_max = float(scenarios_df["accel_magnitude"].quantile(0.99))
    demand_series = scenarios_df["demand"]

    logger.info(f"  Rango accel_magnitude: [{accel_ref_min:.4f}, {accel_ref_max:.4f}] m/s²")
    logger.info(f"  Rango demand:          [{demand_series.min():.1f}, {demand_series.max():.1f}] veh/intervalo")

    # ── Pasos 2-4: Generar un .rou.xml por escenario ──────────────────────
    logger.info("─" * 60)
    logger.info(f"PASOS 2-4 — Generando {len(scenarios_df)} archivos .rou.xml")
    logger.info("─" * 60)

    generated_files: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in scenarios_df.iterrows():
        scenario_num = int(idx) + 1
        try:
            # Paso 2: Mapeo Krauß
            krauss = map_to_krauss(
                accel_magnitude=float(row["accel_magnitude"]),
                jerk=float(row["jerk"]),
                accel_ref_min=accel_ref_min,
                accel_ref_max=accel_ref_max,
            )

            # Factor de densidad vehicular basado en demanda
            demand_factor = _compute_demand_factor(
                demand=float(row["demand"]),
                demand_series=demand_series,
            )

            # Paso 3: Generar XML
            xml_root = generate_route_xml(
                scenario_idx=scenario_num,
                krauss=krauss,
                flow_records=flow_records,
                demand_factor=demand_factor,
                city_tag=city_tag,
                rng=np.random.default_rng(random_seed + scenario_num),
            )

            # Paso 4: Exportar con pretty-print
            out_path = export_scenario(
                xml_root=xml_root,
                scenario_idx=scenario_num,
                city_tag=city_tag,
                output_dir=output_dir,
            )
            generated_files.append(out_path)

            if scenario_num % 10 == 0 or scenario_num == 1:
                logger.info(
                    f"  [{scenario_num:>3}/{len(scenarios_df)}] "
                    f"{out_path.name} | "
                    f"accel={krauss.accel:.2f} decel={krauss.decel:.2f} "
                    f"σ={krauss.sigma:.2f} τ={krauss.tau:.2f}s "
                    f"(demand_factor={demand_factor:.2f})"
                )

        except Exception as exc:
            logger.error(f"  [escenario {scenario_num:03d}] Error — {exc} (saltando)")
            continue

    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║               GENERACIÓN COMPLETADA                     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Archivos generados: {len(generated_files)} / {len(scenarios_df)}")
    logger.info(f"  Directorio de salida: {output_dir}")

    return generated_files


# ===========================================================================
# PUNTO DE ENTRADA
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Genera archivos .rou.xml de SUMO desde escenarios de Vine Copula."
    )
    parser.add_argument(
        "--n-scenarios", type=int, default=N_SCENARIOS_PHASE1,
        help=f"Número de escenarios a generar (default: {N_SCENARIOS_PHASE1})"
    )
    parser.add_argument(
        "--city", type=str, default="Hangzhou",
        help="Carpeta de ciudad en data/raw/macro_traffic/ (default: Hangzhou)"
    )
    parser.add_argument(
        "--flow-file", type=str, default="anon_4_4_hangzhou_real.json",
        help="Nombre del archivo de flujo preferido"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla de aleatoriedad (default: 42)"
    )
    args = parser.parse_args()

    files = run_route_generation(
        n_scenarios=args.n_scenarios,
        city=args.city,
        flow_filename=args.flow_file,
        random_seed=args.seed,
    )
