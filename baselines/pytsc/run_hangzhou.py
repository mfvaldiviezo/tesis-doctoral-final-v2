import sys
import os

# Asegurar que pytsc está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "pytsc")))

from pytsc.run_controllers import evaluate_controllers

if __name__ == "__main__":
    # Vamos a evaluar el modelo MARL SOTA provisto por PyTSC (agent.th)
    controllers = ["rl"]

    hours = 0.2  # 12 minutos (720 segundos) para empatar con la evaluación unifilar
    add_controller_args = {
        "fixed_time": {"green_time": 25},
        "sotl": {"mu": 7, "theta": 5, "phi_min": 5},
    }
    
    add_env_args = {
        "misc": {
            "return_agent_stats": True,
            "return_lane_stats": True,
        },
        "sumo": {
            "render": False,
            "sumo_config_file": "hangzhou_normal.sumocfg",
        },
    }

    print("=========================================")
    print("🚦 Evaluando Tráfico NORMAL (Ideal) SOTA")
    print("=========================================")
    evaluate_controllers(
        scenario="hangzhou",
        simulator_backend="sumo",
        controllers=controllers,
        output_folder="normal_traffic",
        hours=hours,
        add_env_args=add_env_args,
        add_controller_args=add_controller_args,
        profile=False,
    )
    
    print("\n=========================================")
    print("🔥 Evaluando Tráfico LATAM (Caótico) SOTA")
    print("=========================================")
    add_env_args["sumo"]["sumo_config_file"] = "hangzhou_latam.sumocfg"
    evaluate_controllers(
        scenario="hangzhou",
        simulator_backend="sumo",
        controllers=controllers,
        output_folder="latam_traffic",
        hours=hours,
        add_env_args=add_env_args,
        add_controller_args=add_controller_args,
        profile=False,
    )
    
    print("\n✅ Evaluación completada. Revisa baselines/pytsc/pytsc/results/sumo/hangzhou/")
