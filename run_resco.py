import os
import subprocess
import sys

def run_resco_experiment(scenario_type="ideal", algorithm="ipql"):
    """
    Ejecuta el benchmark RESCO con el tráfico ideal o LATAM.
    """
    # Cambiar al directorio raiz de RESCO
    resco_dir = os.path.join("baselines", "RESCO")
    
    # Seleccionamos la ruta dependiendo del escenario
    # En RESCO, la ruta de los escenarios asume ./resco_benchmark/environments/...
    route_file = "hangzhou.rou.xml" if scenario_type == "ideal" else "hangzhou_latam.rou.xml"
    
    if scenario_type == "latam":
        # Aseguramos que exista una ruta falsa con latam en el nombre para activar el caos
        route_latam = os.path.join(resco_dir, "resco_benchmark", "environments", "hangzhou", "hangzhou_latam.rou.xml")
        route_ideal = os.path.join(resco_dir, "resco_benchmark", "environments", "hangzhou", "hangzhou.rou.xml")
        if not os.path.exists(route_latam):
            import shutil
            shutil.copy(route_ideal, route_latam)
            
    print(f"\n{'='*50}")
    print(f"🚦 Evaluando Tráfico {scenario_type.upper()} con RESCO SOTA ({algorithm})")
    print(f"{'='*50}")
    
    # Normalizar nombres CLI → nombres internos de RESCO
    ALGO_ALIAS = {
        "colight":      "CoSLight",
        "coslight":     "CoSLight",
        "ippo":         "IPPO",
        "fixed":        "FIXED",
        "maxpressure":  "MAXPRESSURE",
        "maxwave":      "MAXWAVE",
        "idqn":         "IDQN",
        "mplight":      "MPLight",
    }
    algorithm = ALGO_ALIAS.get(algorithm.lower(), algorithm)

    cmd = [
        sys.executable, "resco_benchmark/main.py",
        "@hangzhou",
        f"@{algorithm}",
        f"route:{route_file}",
        "episodes:1",
        "gui:False",
        "libsumo:False",
        "script_launcher:False",
        "save_console_log:False",
        "latam_chaos:True" if scenario_type == "latam" else "latam_chaos:False"
    ]
    
    try:
        # Ejecutar
        env = os.environ.copy()
        # Agregar directorio actual y directorio de RESCO para que import resco_benchmark funcione
        env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + os.path.abspath(resco_dir)
        
        process = subprocess.Popen(
            cmd,
            cwd=resco_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        if process.returncode != 0:
            sys.exit(process.returncode)
    except Exception as e:
        print(f"Error al ejecutar: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="IPPO", help="Algoritmo a evaluar (IPPO, CoLight, MPLight, FIXED)")
    parser.add_argument("--scenario", type=str, default="ideal", choices=["ideal", "latam"], help="Tipo de tráfico")
    
    args = parser.parse_args()
    run_resco_experiment(args.scenario, args.algo)
