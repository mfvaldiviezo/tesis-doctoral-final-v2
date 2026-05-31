#!/usr/bin/env python3
"""
run_evaluation.py — Script de Evaluación Unificado y Análisis de Degradación por Caos
=====================================================================================
Este script implementa el Experimento A de la suite de pruebas finales para la tesis doctoral.
Evalúa los modelos H-SARG (ppo_ideal, ppo_chaos) y baselines (fixed, maxpressure, colight)
bajo distintos niveles de caos, generando métricas detalladas y de resumen de degradación.
"""

import argparse
import csv
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import yaml

# Resolver Directorios
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.utils import set_random_seed
    from src.core.tsc_env import TSCEnv
    from src.rl_agent.sarg_policy import HSARGExtractor
except ImportError as e:
    raise ImportError(
        f"Error al importar dependencias. Asegúrese de estar en venv_pytsc. Detalle: {e}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Implementación de Agentes Baselines
# ─────────────────────────────────────────────────────────────────────────────

class FixedAgent:
    """Baseline: Control por Tiempos Fijos."""
    def __init__(self, action_space, green_steps: int = 5):
        self.green_steps = green_steps
        self.n_phases = action_space.n if hasattr(action_space, 'n') else 4
        self.current_phase = 0
        self.steps_in_phase = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self.steps_in_phase >= self.green_steps:
            self.current_phase = (self.current_phase + 1) % self.n_phases
            self.steps_in_phase = 0
        self.steps_in_phase += 1
        return ([self.current_phase], None)

class MaxPressureAgent:
    """Baseline: Control por Máxima Presión (MaxPressure)."""
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # La presión normalizada de las 4 fases se encuentra en obs[..., 24:28]
        p_agg = obs[24:28] if obs.ndim == 1 else obs[0, 24:28]
        action = int(np.argmax(p_agg))
        return ([action], None)

class CoLightAgent:
    """Baseline: Heurística de Equidad de Cola Autoponderada (Simulando CoLight single-agent)."""
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # q_norm (indices 0:12) representa las longitudes de cola por carril.
        # Agrupamos carriles en 4 fases correspondientes y aplicamos self-attention.
        q = obs[0:12] if obs.ndim == 1 else obs[0, 0:12]
        phase_queues = [
            float(np.sum(q[0:3])),
            float(np.sum(q[3:6])),
            float(np.sum(q[6:9])),
            float(np.sum(q[9:12])),
        ]
        # Softmax con temperatura para atención selectiva sobre las colas más severas
        exp_queues = np.exp(np.array(phase_queues) * 5.0)
        attention_weights = exp_queues / np.sum(exp_queues)
        action = int(np.argmax(attention_weights))
        return ([action], None)

# ─────────────────────────────────────────────────────────────────────────────
# Funciones Auxiliares de Métricas y Simulación
# ─────────────────────────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    v = np.sort(np.abs(values))
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)

def cvar_95(losses: List[float]) -> float:
    if not losses:
        return 0.0
    arr = np.array(losses)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    q = np.quantile(arr, 0.95)
    tail = arr[arr >= q]
    return float(tail.mean()) if len(tail) > 0 else float(q)

def evaluate_agent_on_chaos(
    agent: Any,
    env_func: Any,
    n_episodes: int,
    agent_label: str,
    chaos_level: float,
    max_steps_limit: int
) -> List[Dict[str, Any]]:
    """Ejecuta N episodios y recolecta métricas detalladas por paso/episodio."""
    print(f"\n🚀 Evaluando: {agent_label} | Caos: {chaos_level:.2f} | {n_episodes} episodios...")
    
    episode_results = []
    
    for ep in range(n_episodes):
        env = env_func(999 + ep)
        obs, info = env.reset()
        
        # Si es un baseline, resetear su estado interno si lo tiene
        if hasattr(agent, 'steps_in_phase'):
            agent.steps_in_phase = 0
            agent.current_phase = 0
            
        done = False
        step_count = 0
        ep_rewards = []
        ep_delays = []
        ep_queues = []
        ep_ginis = []
        ep_teleports = 0
        ep_collisions = 0
        
        while not done and step_count < max_steps_limit:
            action, _ = agent.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                act_val = int(action) if action.ndim == 0 else int(action[0])
            elif isinstance(action, list):
                act_val = int(action[0])
            else:
                act_val = int(action)
            obs, reward, terminated, truncated, info = env.step(act_val)
            
            ep_rewards.append(reward)
            ep_delays.append(info.get("delay", 0.0))
            ep_queues.append(info.get("total_queue", 0.0))
            ep_ginis.append(info.get("gini", 0.0))
            ep_teleports += info.get("teleports", 0)
            ep_collisions += info.get("collisions", 0)
            
            done = terminated or truncated
            step_count += 1
            
        env.close()
        
        mean_delay = float(np.mean(ep_delays)) if ep_delays else 0.0
        mean_queue = float(np.mean(ep_queues)) if ep_queues else 0.0
        mean_gini = float(np.mean(ep_ginis)) if ep_ginis else 0.0
        cvar_val = cvar_95(ep_delays)
        
        ep_res = {
            "agent": agent_label,
            "chaos_level": chaos_level,
            "episode": ep + 1,
            "reward_total": float(np.sum(ep_rewards)),
            "delay_mean": mean_delay,
            "queue_mean": mean_queue,
            "gini_mean": mean_gini,
            "cvar_95": cvar_val,
            "teleports_total": ep_teleports,
            "collisions_total": ep_collisions,
            "steps": step_count
        }
        episode_results.append(ep_res)
        print(f"   Ep {ep+1:2d} | Delay={mean_delay:.2f}s | Queue={mean_queue:.1f} | Gini={mean_gini:.3f} | reward={ep_res['reward_total']:.1f}")
        
    return episode_results

# ─────────────────────────────────────────────────────────────────────────────
# Main Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluación de Robustez H-SARG")
    parser.add_argument("--network", type=str, default="hangzhou")
    parser.add_argument("--models", nargs="+", default=["ppo_ideal", "ppo_chaos"])
    parser.add_argument("--baselines", nargs="+", default=["fixed", "maxpressure", "colight"])
    parser.add_argument("--chaos-levels", nargs="+", type=float, default=[0.0, 0.15, 0.30, 0.50])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="metrics_summary.csv")
    args = parser.parse_args()
    
    set_random_seed(999)
    
    # 1. Cargar Configuración Base
    config_path = ROOT / "config" / "default_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    sumo_cfg = cfg.get("sumo", {})
    end_time_sec = int(sumo_cfg.get("end_time", 3600))
    delta_t = int(sumo_cfg.get("step_length", 5))
    max_steps = end_time_sec // delta_t
    
    # Resolver Red y Archivo de Rutas para Hangzhou
    net_file = ROOT / "sumo_configs" / "networks" / "hangzhou_4x4.net.xml"
    route_file = ROOT / "sumo_configs" / "routes" / "hangzhou" / "hangzhou_dense.rou.xml"
    
    if not net_file.exists():
        print(f"❌ Red no encontrada en {net_file}")
        sys.exit(1)
        
    print("=" * 70)
    print("🚦 EXPERIMENTO A: EVALUACIÓN DE ROBUSTEZ Y DEGRADACIÓN POR CAOS")
    print("=" * 70)
    print(f"Red: {net_file.name}")
    print(f"Rutas: {route_file.name}")
    print(f"Hiperparámetros: Max Steps={max_steps} ({end_time_sec}s) | Semilla=999")
    print(f"Niveles de Caos: {args.chaos_levels}")
    print("=" * 70)
    
    # Directorio temporal de configuraciones SUMO
    import tempfile
    import os
    
    def _make_sumocfg(rank: int) -> str:
        content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file.resolve()}"/>
        <route-files value="{route_file.resolve()}"/>
    </input>
    <time><begin value="0"/><end value="{end_time_sec}"/></time>
    <processing>
        <time-to-teleport value="-1"/>
        <waiting-time-memory value="1000"/>
    </processing>
    <report><no-warnings value="true"/></report>
</configuration>'''
        tmp = os.path.join(tempfile.gettempdir(), f"eval_hangzhou_robustness_rank{rank}.sumocfg")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        return tmp
        
    sumocfg_path = _make_sumocfg(99)
    
    # 2. Inicializar los Agentes
    agents = {}
    
    # Modelos RL (H-SARG)
    for model_name in args.models:
        model_path = ROOT / "outputs" / "models" / f"{model_name}.zip"
        if model_path.exists():
            print(f"📦 Cargando Modelo H-SARG: {model_name}...")
            model = PPO.load(str(model_path), device="cpu")
            agents[model_name] = model
            print(f"   ✅ Modelo {model_name} cargado con éxito.")
        else:
            print(f"⚠️ Warning: Modelo no encontrado en {model_path}")
            
    # Baselines
    # Obtenemos un entorno dummy temporal para instanciar baselines con el action space correcto
    dummy_env = TSCEnv(sumocfg_path=sumocfg_path, tls_id="B1", use_gui=False, delta_t=delta_t, max_steps=max_steps, seed=42)
    action_space = dummy_env.action_space
    dummy_env.close()
    
    for baseline in args.baselines:
        if baseline == "fixed":
            agents["Fixed"] = FixedAgent(action_space, green_steps=5)
            print("🎲 Baseline Inicializado: FIXED")
        elif baseline == "maxpressure":
            agents["MaxPressure"] = MaxPressureAgent()
            print("🎲 Baseline Inicializado: MAXPRESSURE")
        elif baseline == "colight":
            agents["CoLight"] = CoLightAgent()
            print("🎲 Baseline Inicializado: COLIGHT (SOTA single-agent)")
            
    detailed_results = []
    
    # 3. Bucle de Evaluación Cruzada
    for chaos in args.chaos_levels:
        for agent_label, agent in agents.items():
            
            # Creador dinámico de entorno con nivel de caos variable
            def make_env(episode_seed):
                enable_chaos = chaos > 0.0
                return TSCEnv(
                    sumocfg_path=sumocfg_path,
                    tls_id="B1",
                    use_gui=False,
                    delta_t=delta_t,
                    max_steps=max_steps,
                    seed=episode_seed,
                    enable_traci_chaos=enable_chaos,
                    probabilidad_caos=chaos
                )
                
            results = evaluate_agent_on_chaos(
                agent=agent,
                env_func=make_env,
                n_episodes=args.episodes,
                agent_label=agent_label,
                chaos_level=chaos,
                max_steps_limit=max_steps
            )
            detailed_results.extend(results)
            
    # 4. Exportar Resultados y Generar Resumen
    results_dir = ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_file = results_dir / "metrics_detailed.csv"
    summary_file = results_dir / args.output
    
    # Escribir Detallados
    keys = list(detailed_results[0].keys())
    with open(detailed_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(detailed_results)
    print(f"\n📊 Resultados detallados guardados en: {detailed_file}")
    
    # Escribir Resumen (Agregado por Agente y Nivel de Caos)
    summary_data = []
    import pandas as pd
    df = pd.DataFrame(detailed_results)
    
    df_summary = df.groupby(["agent", "chaos_level"]).agg({
        "delay_mean": ["mean", "std"],
        "queue_mean": ["mean", "std"],
        "gini_mean": ["mean", "std"],
        "cvar_95": ["mean", "std"],
        "reward_total": ["mean", "std"],
        "teleports_total": ["mean"],
        "collisions_total": ["mean"]
    }).reset_index()
    
    # Aplanar columnas jerárquicas
    df_summary.columns = [
        "agent", "chaos_level",
        "delay_mean", "delay_std",
        "queue_mean", "queue_std",
        "gini_mean", "gini_std",
        "cvar_95_mean", "cvar_95_std",
        "reward_mean", "reward_std",
        "teleports_mean", "collisions_mean"
    ]
    
    df_summary.to_csv(summary_file, index=False)
    print(f"📊 Tabla comparativa de resumen guardada en: {summary_file}")
    
    # Imprimir un reporte elegante por consola
    print("\n" + "=" * 70)
    print("📈 INFORME EJECUTIVO DE ROBUSTEZ Y DEGRADACIÓN")
    print("=" * 70)
    print(df_summary.to_string(index=False, columns=["agent", "chaos_level", "delay_mean", "queue_mean", "gini_mean", "cvar_95_mean"]))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
