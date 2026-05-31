#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_marl.py — Script Único de Evaluación MARL para Tesis Doctoral
=======================================================================
Red: Hangzhou 4×4  |  Benchmark: RESCO MARL

Uso:
    python evaluate_marl.py                          # Suite completa
    python evaluate_marl.py --skip-existing          # Salta los ya ejecutados
    python evaluate_marl.py --report-only            # Solo genera reporte
    python evaluate_marl.py --algos FIXED IPPO       # Subconjunto
    python evaluate_marl.py --scenarios ideal        # Un escenario

Métricas implementadas (Capítulo 4.3.2 de la tesis):
    Eficiencia:   Throughput/step, Avg Queue Length
    Equidad:      Gini temporal, Gini final por semáforo
    Riesgo:       CVaR95 de la distribución de colas
    Resiliencia:  Degradación porcentual ideal → caótico
"""

import os
import sys
import json
import subprocess
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
RESCO_DIR    = PROJECT_ROOT / "baselines" / "RESCO"
RESULTS_DIR  = RESCO_DIR / "results"
REPORT_DIR   = PROJECT_ROOT / "benchmark_reports"
REPORT_DIR.mkdir(exist_ok=True)

# Añadir framework al path para importar métricas
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from tsc_framework.src.core.reward import gini_coefficient, cvar_calculation
    EQUITY_AVAILABLE = True
except ImportError:
    EQUITY_AVAILABLE = False
    # Fallbacks locales si el framework no está disponible
    def gini_coefficient(values):
        v = np.asarray(values, dtype=float)
        v = v[v >= 0]
        if len(v) <= 1 or np.sum(v) == 0:
            return 0.0
        s = np.sort(v); n = len(s); i = np.arange(1, n+1)
        return float(np.clip((2*np.sum(i*s))/(n*np.sum(s)) - (n+1)/n, 0, 1))

    def cvar_calculation(losses, alpha=0.95):
        l = np.asarray(losses, dtype=float)
        if len(l) == 0: return 0.0, 0.0
        var = np.percentile(l, alpha*100)
        tail = l[l >= var]
        return float(var), float(np.mean(tail)) if len(tail) else float(var)

# ─── Configuración de experimentos ────────────────────────────────────────────
ALGORITHMS = [
    ("FIXED",       "Fixed Time",       "Línea base estática"),
    ("MAXPRESSURE", "Max Pressure",     "Heurística clásica"),
    ("IPPO",        "IPPO",             "Deep RL independiente"),
    ("CoLight",     "CoSLight (SOTA)",  "Graph-Attention MARL"),
]

SCENARIOS = [
    ("ideal", "Tráfico Ideal"),
    ("latam", "Tráfico LATAM (Caótico)"),
]

TIMEOUT_SECONDS = {
    "FIXED":       1800,
    "MAXPRESSURE": 1800,
    "IPPO":        1800,
    "CoLight":     1800,
}

# ─── Colores ANSI ─────────────────────────────────────────────────────────────
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_RED    = "\033[91m"
C_CYAN   = "\033[96m"
C_BOLD   = "\033[1m"
C_RESET  = "\033[0m"

def cprint(color, text):
    print(f"{color}{text}{C_RESET}")

def box(title, width=72):
    print(f"\n{C_BOLD}{'═'*width}{C_RESET}")
    print(f"{C_BOLD}  {title}{C_RESET}")
    print(f"{C_BOLD}{'═'*width}{C_RESET}")

# ─── Ejecución de experimento ─────────────────────────────────────────────────
def run_experiment(algo: str, scenario: str) -> dict:
    """Lanza run_resco.py como subprocess y captura métricas."""
    timeout = TIMEOUT_SECONDS.get(algo, 600)
    cprint(C_CYAN, f"  ▶ Ejecutando {algo} | {scenario}  (timeout={timeout}s)...")
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"]      = str(PROJECT_ROOT) + os.pathsep + str(RESCO_DIR)
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "run_resco.py"),
         "--algo", algo, "--scenario", scenario],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=str(PROJECT_ROOT),
        encoding="utf-8", errors="replace",
    )

    episode_reward = None
    stdout_lines = []
    stderr_lines = []
    try:
        stdout_data, stderr_data = proc.communicate(timeout=timeout)
        stdout_lines = stdout_data.splitlines()
        stderr_lines = stderr_data.splitlines()
        for line in stdout_lines:
            if "Episode Reward:" in line:
                try:
                    episode_reward = float(
                        line.split("Episode Reward:")[-1].strip().rstrip(",)."))
                except ValueError:
                    pass
    except subprocess.TimeoutExpired:
        proc.kill()
        cprint(C_RED, f"    ⚠️  TIMEOUT ({timeout}s) — el proceso fue terminado")

    elapsed  = round(time.time() - t0, 1)
    success  = proc.returncode == 0

    if success:
        cprint(C_GREEN, f"    ✅ OK en {elapsed}s  |  Episode Reward = {episode_reward}")
        if algo == "MAXPRESSURE":
            cprint(C_YELLOW, "    [DEBUG] MAXPRESSURE output completo:")
            for ln in (stdout_lines + stderr_lines):
                if ln.strip():
                    print(f"       {ln}")
    else:
        cprint(C_RED, f"    ❌ Error (código {proc.returncode})")
        # Mostrar todas las líneas de stdout y stderr para diagnóstico
        for ln in (stdout_lines + stderr_lines):
            if ln.strip():
                print(f"       {ln}")

    return {
        "algo": algo, "scenario": scenario,
        "success": success, "elapsed_s": elapsed,
        "episode_reward": episode_reward, "returncode": proc.returncode,
    }


# ─── Buscar métricas guardadas ─────────────────────────────────────────────────
def find_metrics(algo: str, scenario: str) -> Optional[dict]:
    """Busca el unified_metrics_ep*.json más reciente para algo+scenario."""
    ALGO_KEY = {"CoLight": "coslight", "IPPO": "ippo",
                "FIXED": "fixed", "MAXPRESSURE": "maxpressure"}
    algo_key = ALGO_KEY.get(algo, algo.lower())
    best_mtime, best_data = 0, None

    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir(): continue
        name = run_dir.name.lower()
        if algo_key not in name: continue
        # CRITICAL: 'latam' appears in ALL dirs as 'latamchaos@false' or 'latamchaos@true'
        # Must check the explicit flag value, not just presence of substring
        is_latam_run = "latamchaos@true" in name
        if scenario == "latam" and not is_latam_run: continue
        if scenario == "ideal" and is_latam_run:     continue
        for mf in run_dir.rglob("unified_metrics_ep*.json"):
            mt = mf.stat().st_mtime
            if mt > best_mtime:
                try:
                    best_data  = json.loads(mf.read_text(encoding="utf-8"))
                    best_mtime = mt
                except Exception:
                    pass
    return best_data


# ─── Post-análisis: recalcula Gini / CVaR desde el JSON ───────────────────────
def enrich_metrics(m: dict) -> dict:
    """Asegura que todos los campos de equidad están presentes."""
    if m is None:
        return {}
    # Si el JSON ya fue generado con el nuevo hook, tiene todo.
    # Si proviene de una ejecución antigua, calculamos lo que falta.
    if "gini_temporal" not in m:
        m["gini_temporal"] = "N/D"
    if "cvar95_queue" not in m:
        m["cvar95_queue"] = "N/D"
    if "gini_final" not in m:
        m["gini_final"] = "N/D"
    return m


# ─── Generación de tabla comparativa ──────────────────────────────────────────
def build_report(all_results: list[dict]) -> str:
    W = 110
    SEP = "─" * W
    lines = []

    # ── Encabezado ──
    lines += [
        "",
        "═"*W,
        f"{'EVALUACIÓN DE RESILIENCIA MARL — RED HANGZHOU 4×4':^{W}}",
        f"{'Tesis Doctoral | Capítulo 4.3.2 — Métricas: Eficiencia, Equidad (Gini) y Riesgo (CVaR95)':^{W}}",
        f"{'Generado: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{W}}",
        "═"*W,
    ]

    # ── Tabla de resultados completa ──
    lines += [
        "",
        f"  {'ALGORITMO':<18} {'ESCENARIO':<20} {'THROUGHPUT/s':>13} {'AVG QUEUE':>11} "
        f"{'GINI_temp':>10} {'CVaR95':>9} {'GINI_final':>11} {'TIEMPO(s)':>10} {'OK':>4}",
        SEP,
    ]

    algo_order = [a[0] for a in ALGORITHMS]
    grouped: dict[str, dict] = {}
    for r in all_results:
        grouped.setdefault(r["algo"], {})[r["scenario"]] = r

    for algo in algo_order:
        if algo not in grouped: continue
        for scen in ["ideal", "latam"]:
            r = grouped[algo].get(scen)
            if r is None:
                lines.append(f"  {algo:<18} {'  '+scen:<20} {'—':>13} {'—':>11} {'—':>10} {'—':>9} {'—':>11} {'—':>10} {'?':>4}")
                continue
            m = enrich_metrics(find_metrics(algo, scen))

            def fmt(v, d=4): return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)
            tput  = fmt(m.get("throughput_per_step",  "N/D"), 4)
            avgq  = fmt(m.get("avg_queue_length",     "N/D"), 3)
            gini  = fmt(m.get("gini_temporal",        "N/D"), 4)
            cvar  = fmt(m.get("cvar95_queue",         "N/D"), 3)
            ginif = fmt(m.get("gini_final",           "N/D"), 4)
            ok    = "✓" if r["success"] else "✗"
            slabel = "Ideal" if scen == "ideal" else "LATAM Caótico"
            lines.append(
                f"  {algo:<18} {slabel:<20} {tput:>13} {avgq:>11} {gini:>10} {cvar:>9} {ginif:>11} {r['elapsed_s']:>10} {ok:>4}"
            )
        lines.append(SEP)

    # ── Análisis de degradación (resiliencia) ──
    lines += [
        "",
        f"  {'📊 ANÁLISIS DE DEGRADACIÓN  (Ideal → LATAM Caótico)':}",
        "─"*75,
        f"  {'ALGORITMO':<18} {'Δ Throughput':>14} {'Δ AvgQueue':>12} {'Δ Gini_temp':>13} {'Δ CVaR95':>10} {'RESILIENCIA':>13}",
        "─"*75,
    ]

    for algo in algo_order:
        mi = find_metrics(algo, "ideal")
        ml = find_metrics(algo, "latam")
        if mi and ml:
            def pct(a, b, key):
                vi, vl = a.get(key, 0), b.get(key, 0)
                if not isinstance(vi, (int, float)) or vi == 0: return float("nan")
                return ((vl - vi) / abs(vi)) * 100

            dt  = pct(mi, ml, "throughput_per_step")
            dq  = pct(mi, ml, "avg_queue_length")
            dgi = pct(mi, ml, "gini_temporal")
            dc  = pct(mi, ml, "cvar95_queue")

            def fmtd(v): return f"{v:+.1f}%" if not np.isnan(v) else "N/D"

            # Clasificar resiliencia por throughput
            if np.isnan(dt):
                resil = "Sin datos"
            elif abs(dt) < 10:
                resil = "Alta  🟢"
            elif abs(dt) < 25:
                resil = "Media 🟡"
            else:
                resil = "Baja  🔴"

            lines.append(
                f"  {algo:<18} {fmtd(dt):>14} {fmtd(dq):>12} {fmtd(dgi):>13} {fmtd(dc):>10} {resil:>13}"
            )
        else:
            lines.append(f"  {algo:<18} {'Datos incompletos (ejecutar ambos escenarios)':>69}")

    lines += ["─"*75, ""]

    # ── Ranking por métrica ──
    lines += ["  🏆 RANKING POR EFICIENCIA (Throughput — mayor es mejor)"]
    lines += ["─"*50]
    rank_data = []
    for algo in algo_order:
        for scen in ["ideal", "latam"]:
            m = find_metrics(algo, scen)
            if m and isinstance(m.get("throughput_per_step"), (int, float)):
                rank_data.append((algo, scen, m["throughput_per_step"], m.get("gini_temporal","?"), m.get("cvar95_queue","?")))
    rank_data.sort(key=lambda x: x[2], reverse=True)
    for i, (a, s, t, g, c) in enumerate(rank_data, 1):
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
        lines.append(f"  {medal}  {a:<15} [{s:<6}]  Throughput={t:.4f}  Gini={g}  CVaR95={c}")
    lines += ["─"*50, ""]

    # ── Equidad: Ranking Gini ──
    lines += ["  ⚖️  RANKING POR EQUIDAD (Gini_temporal — menor es mejor)"]
    lines += ["─"*50]
    gini_data = []
    for algo in algo_order:
        for scen in ["ideal", "latam"]:
            m = find_metrics(algo, scen)
            if m and isinstance(m.get("gini_temporal"), (int, float)):
                gini_data.append((algo, scen, m["gini_temporal"]))
    gini_data.sort(key=lambda x: x[2])
    for i, (a, s, g) in enumerate(gini_data, 1):
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
        lines.append(f"  {medal}  {a:<15} [{s:<6}]  Gini = {g:.5f}")
    lines += ["─"*50, ""]

    # ── Notas metodológicas ──
    lines += [
        "═"*W,
        "  📋 NOTAS METODOLÓGICAS (Capítulo 4.3.2):",
        "  • Throughput/step  : Vehículos que completan el recorrido por paso de simulación (↑ mejor)",
        "  • Avg Queue        : Longitud media de colas por semáforo por paso (↓ mejor)",
        "  • Gini_temporal    : Coeficiente de Gini sobre colas a lo largo del tiempo ∈ [0,1] (↓ = más equitativo)",
        "  • Gini_final       : Gini de la distribución de colas finales entre semáforos (↓ = más uniforme)",
        "  • CVaR95(cola)     : Valor esperado del 5% peor de la distribución de colas (↓ mejor)",
        "  • Δ Degradación    : ((caótico - ideal) / |ideal|) × 100  — resiliencia al estrés LATAM",
        "  • Resiliencia      : Alta |Δ|<10%, Media |Δ|<25%, Baja |Δ|≥25%",
        "  ⚠️  Episode Reward NO se incluye en rankings (escala incomparable entre FIXED/IPPO y CoSLight)",
        "═"*W,
    ]

    return "\n".join(lines)


# ─── Guardar reporte ──────────────────────────────────────────────────────────
def save_report(table: str, all_results: list[dict]):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt  = REPORT_DIR / f"marl_eval_{ts}.txt"
    jsonf = REPORT_DIR / f"marl_eval_{ts}_raw.json"
    txt.write_text(table, encoding="utf-8")
    jsonf.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    cprint(C_GREEN, f"\n  💾 Reporte guardado en:\n     {txt}\n     {jsonf}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluación MARL completa — Hangzhou 4×4 (Tesis Doctoral)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Salta experimentos si ya tienen unified_metrics")
    parser.add_argument("--report-only", action="store_true",
                        help="Solo genera reporte con resultados existentes")
    parser.add_argument("--algo", type=str, default="ALL",
                        help="Ejecutar solo un algoritmo en particular (ej: MAXPRESSURE)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        choices=["ideal", "latam"],
                        help="Subset de escenarios")
    args = parser.parse_args()

    algos_to_run = [a[0] for a in ALGORITHMS]
    if args.algo != "ALL":
        algos_to_run = [a for a in algos_to_run if a.upper() == args.algo.upper()]
        if not algos_to_run:
            print(f"Algoritmo '{args.algo}' no reconocido. Opciones: {[a[0] for a in ALGORITHMS]}")
            sys.exit(1)

    scenarios_to_run = args.scenarios or ["ideal", "latam"]

    box("🚦 EVALUACIÓN MARL — HANGZHOU 4×4 — TESIS DOCTORAL")
    print(f"  Algoritmos  : {algos_to_run}")
    print(f"  Escenarios  : {scenarios_to_run}")
    print(f"  Equity (Gini/CVaR): {'✅ Activo' if EQUITY_AVAILABLE else '⚠️ Modo fallback'}")
    print(f"  Resultados  : {RESULTS_DIR}")
    print(f"  Reportes    : {REPORT_DIR}")

    all_results: list[dict] = []

    if not args.report_only:
        total   = len(algos_to_run) * len(scenarios_to_run)
        current = 0
        for algo in algos_to_run:
            for scenario in scenarios_to_run:
                current += 1
                box(f"[{current}/{total}]  {algo}  ×  {scenario.upper()}", width=60)

                if args.skip_existing:
                    existing = find_metrics(algo, scenario)
                    if existing:
                        cprint(C_YELLOW, "  ⏩ Resultado existente encontrado — saltando (--skip-existing)")
                        all_results.append({
                            "algo": algo, "scenario": scenario,
                            "success": True, "elapsed_s": 0,
                            "episode_reward": existing.get("episode_reward"),
                            "returncode": 0, "skipped": True,
                        })
                        continue

                result = run_experiment(algo, scenario)
                all_results.append(result)
                time.sleep(3)  # Pausa para que SUMO libere puertos TCP
    else:
        # Modo reporte: poblar all_results con los datos existentes
        for algo in algos_to_run:
            for scenario in scenarios_to_run:
                m = find_metrics(algo, scenario)
                all_results.append({
                    "algo": algo, "scenario": scenario,
                    "success": m is not None, "elapsed_s": 0,
                    "episode_reward": m.get("episode_reward") if m else None,
                    "returncode": 0, "skipped": True,
                })

    # ─── Tabla + reporte ──────────────────────────────────────────────────────
    box("📊 TABLA COMPARATIVA COMPLETA", width=72)
    report = build_report(all_results)
    print(report)
    save_report(report, all_results)

    # ─── Resumen final ────────────────────────────────────────────────────────
    failures = [r for r in all_results if not r.get("success") and not r.get("skipped")]
    if failures:
        cprint(C_RED, f"\n  ⚠️  {len(failures)} experimento(s) fallaron:")
        for f in failures:
            cprint(C_RED, f"       {f['algo']} | {f['scenario']} (código {f.get('returncode')})")
    else:
        cprint(C_GREEN, "\n  ✅ Evaluación completada.")


if __name__ == "__main__":
    main()
