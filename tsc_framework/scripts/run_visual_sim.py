import os
import sys
import traci
from sumolib import checkBinary

# Agregar el path del framework al path de python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.latam_chaos_manager import LatamChaosManager

def main():
    # Usar los archivos reales que contienen la infraestructura y flujos de Quito
    # - net_file tiene soporte de aceras para peatones
    # - route_file define moto_imprudent (motos), micro_imprudent (buses) y ped_imprudent (peatones)
    net_file = r"tsc_framework/sumo_configs/hangzhou/hangzhou_pedestrian.net.xml"
    route_file = r"tsc_framework/experiments/hangzhou_robustness/scenarios/latam_imprudent_drivers.rou.xml"
    
    # Validar que los archivos existan
    if not os.path.exists(net_file) or not os.path.exists(route_file):
        workspace = r"C:\Proyecto_Tesis_Final_V1\traffic_project"
        net_file = os.path.join(workspace, net_file)
        route_file = os.path.join(workspace, route_file)
        
        if not os.path.exists(net_file) or not os.path.exists(route_file):
            print("❌ ERROR: No se pudieron encontrar los archivos de escenario con motos y peatones.")
            print(f"Rutas buscadas:\n - Net: {net_file}\n - Route: {route_file}")
            return

    print("════════════════════════════════════════════════════════════")
    print("🚦 SIMULACIÓN VISUAL LATAM COMPLETA (MOTOS + PEATONES + BUSES)")
    print("════════════════════════════════════════════════════════════")
    print("  • Verdes [Motos]: Motocicletas imprudentes (Lane Splitting / Sublanes)")
    print("  • Naranjas [Buses]: Micros haciendo paradas informales cerca de peatones")
    print("  • Celestes [Peatones]: Peatones cruzando e interactuando con vehículos")
    print("  • Rojos [Autos]: Perfiles reales de conductores imprudentes de Quito")
    print("  • Física de colisión, atascos y bloqueos transversales ACTIVOS")
    print("════════════════════════════════════════════════════════════")

    sumo_binary = checkBinary("sumo-gui")
    
    # Habilitar el modelo de sublanes para que las motocicletas puedan rebasar entre carriles (lane splitting)
    # y activar la física de colisiones real de Quito
    sumo_cmd = [
        sumo_binary,
        "-n", net_file,
        "-r", route_file,
        "--begin", "0",
        "--end", "3600",
        "--delay", "50",                   # Retraso para ver los detalles
        "--lateral-resolution", "0.4",     # MÁGICO: Habilita el sublane model en SUMO para que las motos hagan lane splitting!
        "--collision.action", "teleport",  # Activar choques físicos
        "--collision.mingap-factor", "0",   # Desactivar protección de SUMO
        "--time-to-teleport", "120",        # Penalización por bloqueo total
        "--random", "True",
        "--no-warnings", "True"
    ]
    
    try:
        traci.start(sumo_cmd)
        
        # Gestor de caos que coordina las paradas de las micros y el arrastre de peatones
        chaos_manager = LatamChaosManager(probabilidad_caos=0.4)
        
        print("\n🚀 Simulación visual iniciada con éxito.")
        print("💡 TIP: Presiona 'Play' en la interfaz gráfica de SUMO.")
        print("💡 TIP: Haz zoom en los semáforos o vías para observar a los peatones celestes cruzando y las motos verdes filtrándose entre carriles.")
        
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            chaos_manager.step()
            
            step += 1
            if step % 100 == 0:
                peds_active = len(traci.person.getIDList())
                veh_active = len(traci.vehicle.getIDList())
                print(f"  [Step {step:4d}] Autos/Motos/Buses activos: {veh_active:3d} | Peatones activos: {peds_active:3d}")
                
    except traci.exceptions.FatalTraCIError:
        print("\n✅ Conexión con SUMO finalizada (Ventana cerrada).")
    except Exception as e:
        print(f"\n⚠️ Ocurrió un error: {e}")
    finally:
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    main()