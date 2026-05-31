"""
evaluate.py — Evaluación Doctoral del Agente PPO (Versión Corregida)
====================================================================
Capítulo 4.5 — Tesis Doctoral: Análisis de Rendimiento del Agente

Correcciones aplicadas:
  1. Manejo robusto de unpacking para reset() y step() (SB3 vs Gymnasium).
  2. Extracción profunda de métricas desde el diccionario 'info' (anidado o directo).
  3. Corrección de error de formato en la impresión de tablas.
  4. Fallback seguro si las métricas son nulas.
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Funciones de Métricas
# ─────────────────────────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    """Índice de Gini sobre distribución de esperas."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0: return 0.0
    idx = np.arange(1, n + 1)
    sum_idx_v = np.sum(idx * v)
    sum_v = np.sum(v)
    if sum_v == 0: return 0.0
    return float((2 * sum_idx_v) / (n * sum_v) - (n + 1) / n)


def cvar(losses: List[float], alpha: float = 0.90) -> float:
    """CVaR_alpha: media del (1-alpha)% peor de las pérdidas."""
    if not losses:
        return 0.0
    arr = np.array(losses)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0: return 0.0
    q = np.quantile(arr, alpha)
    tail = arr[arr >= q]
    return float(tail.mean()) if len(tail) > 0 else float(q)


# ─────────────────────────────────────────────────────────────────────────────
# Funciones Auxiliares de Extracción
# ─────────────────────────────────────────────────────────────────────────────

def extract_metric(info: Any, key: str, default: float = 0.0) -> float:
    """
    Extrae una métrica del diccionario info, manejando anidamientos de VecEnv/Monitor.
    Busca en: info[key], info[0][key], info['episode'][key] (si estuviera disponible en step).
    """
    if info is None:
        return default
    
    # Caso 1: info es un dict directo
    if isinstance(info, dict):
        val = info.get(key)
        if val is not None: return float(val)
        
    # Caso 2: info es una lista/tupla (común en VecEnv si no se unwrappea bien)
    if isinstance(info, (list, tuple)) and len(info) > 0:
        first = info[0]
        if isinstance(first, dict):
            val = first.get(key)
            if val is not None: return float(val)
            
    # Caso 3: Búsqueda recursiva simple si hay anidamiento raro
    # (Aunque en step-by-step normalmente está en la raíz o en info[0])
    
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación de un agente sobre N episodios
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(model: Any, env: Any, n_episodes: int, label: str, max_steps_limit: int) -> Dict:
    """Ejecuta N episodios y devuelve diccionario de métricas agregadas."""
    print(f"\n{'─'*55}")
    print(f"  Evaluando: {label} | {n_episodes} episodios")
    print(f"{'─'*55}")

    all_rewards:    List[float] = []
    all_delays:     List[float] = []
    all_waits:      List[float] = []
    all_queues:     List[float] = []
    all_gini:       List[float] = []
    all_teleports:  List[float] = []
    all_collisions: List[float] = []
    episode_losses: List[float] = []

    for ep in range(n_episodes):
        # Reset robusto
        try:
            res = env.reset()
            if isinstance(res, tuple) and len(res) == 2:
                obs, info_reset = res
            else:
                obs = res
                info_reset = {}
        except Exception as e:
            print(f"❌ Error en reset(): {e}")
            try:
                import glob
                log_files = glob.glob(str(ROOT / "*.sumo.log"))
                if log_files:
                    with open(log_files[0], "r", encoding="utf-8", errors="ignore") as f:
                        print(f"--- SUMO LOG ---\n{f.read()}\n----------------")
            except:
                pass
            break
        
        done = False
        ep_reward = 0.0
        ep_delays: List[float] = []
        ep_waits:  List[float] = []
        ep_queues: List[float] = []
        ep_teleports_count = 0
        ep_collisions_count = 0
        steps = 0

        ep_gini_list: List[float] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Step robusto
            try:
                step_res = env.step(action)
                if len(step_res) == 5:
                    obs, reward, terminated, truncated, info = step_res
                    done = terminated or truncated
                elif len(step_res) == 4:
                    obs, reward, done, info = step_res
                else:
                    obs, reward = step_res[0], step_res[1]
                    done = False
                    info = {}
            except Exception as e:
                print(f"⚠️ Error en step(): {e}. Terminando episodio.")
                break
            
            # Acumular reward
            if isinstance(reward, (list, np.ndarray)) and len(reward) > 0:
                ep_reward += float(reward[0])
            else:
                ep_reward += float(reward)

            # Extraer métricas usando la función auxiliar con las keys correctas de tsc_env.py
            delay_val = extract_metric(info, "delay")
            queue_val = extract_metric(info, "total_queue")
            gini_env = extract_metric(info, "gini")
            
            ep_teleports_count += extract_metric(info, "teleports", 0.0)
            ep_collisions_count += extract_metric(info, "collisions", 0.0)
            
            ep_delays.append(delay_val)
            ep_waits.append(delay_val / 12.0)  # Approx average wait per lane
            ep_queues.append(queue_val)
            if gini_env > 0:
                ep_gini_list.append(gini_env)
            
            steps += 1
            if steps > max_steps_limit + 5: 
                break

        # Cálculos finales del episodio
        delay_arr = np.array(ep_delays)
        all_rewards.append(ep_reward)
        
        if len(delay_arr) > 0:
            mean_delay = float(np.mean(delay_arr))
            gini_val = float(np.mean(ep_gini_list)) if ep_gini_list else 0.0
        else:
            mean_delay = 0.0
            gini_val = 0.0
            
        all_delays.append(mean_delay)
        all_gini.append(gini_val)
        all_waits.append(float(np.mean(ep_waits)) if ep_waits else 0.0)
        all_queues.append(float(np.mean(ep_queues)) if ep_queues else 0.0)
        all_teleports.append(ep_teleports_count)
        all_collisions.append(ep_collisions_count)
        episode_losses.append(float(np.sum(delay_arr)))

        print(f"  Ep {ep+1:3d}/{n_episodes} | reward={ep_reward:8.2f} | "
              f"delay={mean_delay:.3f} | queue={all_queues[-1]:.1f} | "
              f"Gini={gini_val:.3f} | tp={ep_teleports_count} | col={ep_collisions_count} | steps={steps}")

    metrics = {
        "label":          label,
        "n_episodes":     n_episodes,
        "reward_mean":    float(np.mean(all_rewards)),
        "reward_std":     float(np.std(all_rewards)),
        "delay_mean":     float(np.mean(all_delays)),
        "delay_std":      float(np.std(all_delays)),
        "wait_mean":      float(np.mean(all_waits)),
        "queue_mean":     float(np.mean(all_queues)),
        "gini_mean":      float(np.mean(all_gini)),
        "gini_std":       float(np.std(all_gini)),
        "teleports_mean": float(np.mean(all_teleports)),
        "collisions_mean":float(np.mean(all_collisions)),
        "cvar90":         cvar(episode_losses, alpha=0.90),
    }
    return metrics


def print_report(results: List[Dict]) -> None:
    """Imprime tabla comparativa de resultados."""
    if not results:
        return

    print(f"\n{'═'*65}")
    print("  REPORTE FINAL — Evaluación Doctoral TSC Framework")
    print(f"{'═'*65}")
    header = f"{'Métrica':<22} " + " ".join(f"{r['label']:>16}" for r in results)
    print(header)
    print("─" * 65)

    metrics_display = [
        ("Reward medio",   "reward_mean"),
        ("Reward std",     "reward_std"),
        ("Delay medio (s)","delay_mean"),
        ("Delay std",      "delay_std"),
        ("Espera media(s)","wait_mean"),
        ("Cola media(veh)","queue_mean"),
        ("Gini (equidad)", "gini_mean"),
        ("Gini std",       "gini_std"),
        ("Teleports/ep",   "teleports_mean"),
        ("Colisiones/ep",  "collisions_mean"),
        ("CVaR₉₀",        "cvar90"),
    ]

    for display_name, key in metrics_display:
        row = f"  {display_name:<20} "
        for r in results:
            val = r.get(key, 0.0)
            # CORRECCIÓN DE FORMATO: Usar f-string directo para evitar errores de especificadores compuestos
            if key in ["reward_mean", "reward_std"]:
                row += f" {val:16.2f}"
            elif key in ["cvar90"]:
                row += f" {val:16.4f}"
            else:
                row += f" {val:16.4f}"
        print(row)

    print(f"{'─'*65}")

    if len(results) == 2:
        ppo = results[0]
        base = results[1]
        if base["delay_mean"] > 0:
            delay_imp = (base["delay_mean"] - ppo["delay_mean"]) / base["delay_mean"] * 100
            print(f"\n  ✅ Mejora PPO vs Baseline:")
            print(f"     Delay:  {delay_imp:+.1f}%")
            if base["gini_mean"] > 0:
                gini_imp = (base["gini_mean"] - ppo["gini_mean"]) / base["gini_mean"] * 100
                print(f"     Gini:   {gini_imp:+.1f}%")

    print(f"\n{'═'*65}\n")


def save_csv(results: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results: return
    keys = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  📄 Métricas guardadas en: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSC Framework — Evaluación Doctoral PPO")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--no-baseline", action="store_true",
                        help="Omitir evaluación baseline (más rápido)")
    parser.add_argument("--seed",       type=int, default=999)
    parser.add_argument("--route-file", type=str, default=None,
                        help="Ruta a un archivo .rou.xml específico (ej. escenarios de estrés)")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("\n" + "═" * 55)
    print("  TSC Framework — Evaluación Post-Entrenamiento")
    print("═" * 55)

    import yaml
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.utils import set_random_seed
    from src.core.tsc_env import TSCEnv
    import tempfile, os

    cfg_path = ROOT / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    net_cfg = cfg.get("network", {}).get("benchmark", {})
    sumo_cfg = cfg.get("sumo", {})
    
    NETWORK = ROOT / net_cfg.get("network_file", "sumo_configs/networks/hangzhou_4x4.net.xml")
    ROUTE_DIR = ROOT / net_cfg.get("route_files_dir", "sumo_configs/routes/hangzhou")
    TLS_ID = net_cfg.get("tls_id", "B1")
    DELTA_T = int(sumo_cfg.get("step_length", 5))
    end_time_sec = int(sumo_cfg.get("end_time", 3600))
    TIME_TO_TELEPORT = int(sumo_cfg.get("time_to_teleport", -1))
    MAX_STEPS = end_time_sec // DELTA_T
    
    latam_feat = cfg.get("latam_features", {})
    lateral_res = latam_feat.get("lateral_resolution", None)

    if args.route_file:
        ROUTE_FILE = str(ROOT / args.route_file)
    else:
        route_files = sorted(Path(ROUTE_DIR).glob("*.rou.xml")) if Path(ROUTE_DIR).exists() else []
        ROUTE_FILE = str(route_files[0]) if route_files else str(ROOT / "sumo_configs/routes/hangzhou/hangzhou_minimal.rou.xml")

    def _make_sumocfg(rank: int) -> str:
        add_path = ROOT / "experiments/hangzhou_robustness/scenarios/latam_infrastructure.add.xml"
        add_files = f'\n        <additional-files value="{add_path}"/>' if add_path.exists() else ''
        content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{NETWORK}"/>
        <route-files value="{ROUTE_FILE}"/>{add_files}
    </input>
    <time><begin value="0"/><end value="{end_time_sec}"/></time>
    <processing>
        <time-to-teleport value="{TIME_TO_TELEPORT}"/>
        {f'<lateral-resolution value="{lateral_res}"/>' if lateral_res else ''}
    </processing>
    <report><no-warnings value="true"/></report>
</configuration>'''
        tmp = os.path.join(ROOT, f"evaluate_sumo_rank{rank}.sumocfg")
        with open(tmp, "w", encoding="utf-8") as f: f.write(content)
        return tmp

    def _make_env(rank: int, use_gui: bool = False):
        def _thunk():
            set_random_seed(args.seed + rank)
            enable_chaos = latam_feat.get("enable_traci_chaos", False)
            return TSCEnv(
                sumocfg_path=_make_sumocfg(rank),
                tls_id=TLS_ID,
                delta_t=DELTA_T,
                max_steps=MAX_STEPS,
                use_gui=use_gui,
                seed=args.seed + rank,
                enable_traci_chaos=enable_chaos,
            )
        return _thunk

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        sys.exit(1)

    print(f"\n  📦 Cargando modelo: {model_path.name}")
    model = PPO.load(str(model_path), device="cpu")
    print(f"  ✅ Modelo cargado")

    results = []

    # Evaluar PPO
    eval_env = VecMonitor(DummyVecEnv([_make_env(rank=900, use_gui=args.render)]))
    ppo_metrics = evaluate_agent(model, eval_env, args.n_episodes, label="PPO", max_steps_limit=MAX_STEPS)
    results.append(ppo_metrics)
    eval_env.close()

    # Evaluar Baseline
    if not args.no_baseline:
        print(f"\n  🎲 Evaluando baseline...")
        base_env = VecMonitor(DummyVecEnv([_make_env(rank=901)]))
        class RandomAgent:
            def __init__(self, action_space): self.action_space = action_space
            def predict(self, obs, deterministic=False):
                act = self.action_space.sample()
                # DummyVecEnv siempre pasa obs con dimensión [n_envs, ...]
                return ([act], None)
        
        random_model = RandomAgent(base_env.action_space)
        base_metrics = evaluate_agent(random_model, base_env, args.n_episodes, label="Baseline", max_steps_limit=MAX_STEPS)
        results.append(base_metrics)
        base_env.close()

    print_report(results)
    save_csv(results, ROOT / "outputs" / "results" / "evaluation_results.csv")
    print("  📊 TensorBoard: tensorboard --logdir outputs/tensorboard\n")

if __name__ == "__main__":
    main()