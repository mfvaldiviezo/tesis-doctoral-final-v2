#!/usr/bin/env python3
"""
Script para generar conductores y peatones imprudentes basados en datos reales de Quito.
Implementa comportamientos específicos LATAM: 
- Motocicletas (lane splitting)
- Transporte público informal (micros)
- Distribución Beta de Imprudencia
- Bloqueo de intersecciones y salto de prioridades
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    logger.info(f"Cargando datos desde {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
    df = pd.read_csv(data_path)
    return df

def analyze_drivers(df: pd.DataFrame) -> dict:
    logger.info("Analizando comportamiento por conductor y aplicando distribución Beta...")
    driver_stats = {}
    
    # 1. Calcular raw scores
    raw_scores = []
    for driver_id in df['driver_id'].unique():
        driver_data = df[df['driver_id'] == driver_id]
        
        avg_accel = driver_data['accel_magnitude'].mean()
        std_accel = driver_data['accel_magnitude'].std()
        max_accel = driver_data['accel_magnitude'].max()
        avg_jerk = abs(driver_data['jerk']).mean()
        max_jerk = abs(driver_data['jerk']).max()
        avg_speed = driver_data['speed'].mean()
        max_speed = driver_data['speed'].max()
        
        raw_score = (
            0.4 * (avg_accel / df['accel_magnitude'].max()) +
            0.3 * (avg_jerk / df['jerk'].abs().max()) +
            0.2 * (std_accel / df['accel_magnitude'].std()) +
            0.1 * (max_accel / df['accel_magnitude'].max())
        )
        
        raw_scores.append((driver_id, raw_score, len(driver_data), avg_accel, std_accel, max_accel, avg_jerk, max_jerk, avg_speed, max_speed))
    
    # Ordenar por raw score
    raw_scores.sort(key=lambda x: x[1])
    n_drivers = len(raw_scores)
    
    # 2. Asignar clusters (Prudentes 30%, Oportunistas 50%, Temerarios 20%)
    for i, data in enumerate(raw_scores):
        driver_id, raw_score, n_samples, avg_accel, std_accel, max_accel, avg_jerk, max_jerk, avg_speed, max_speed = data
        pct = i / n_drivers
        
        if pct < 0.30:
            # Prudentes: 0.0 - 0.3
            final_score = np.random.uniform(0.0, 0.3)
        elif pct < 0.80:
            # Oportunistas: Beta(2, 4) mapeado a 0.3 - 0.7
            final_score = 0.3 + np.random.beta(2, 4) * 0.4
        else:
            # Temerarios: Beta(0.5, 2) mapeado a 0.7 - 1.0
            final_score = 0.7 + np.random.beta(0.5, 2) * 0.3
            final_score = min(1.0, final_score)
            
        driver_stats[driver_id] = {
            'n_samples': n_samples,
            'avg_accel': float(avg_accel),
            'std_accel': float(std_accel),
            'max_accel': float(max_accel),
            'avg_jerk': float(avg_jerk),
            'max_jerk': float(max_jerk),
            'avg_speed': float(avg_speed),
            'max_speed': float(max_speed),
            'imprudence_score': float(final_score)
        }
    
    sorted_drivers = sorted(driver_stats.items(), key=lambda x: x[1]['imprudence_score'], reverse=True)
    return dict(sorted_drivers)

def export_latam_infrastructure(output_path: str):
    logger.info(f"Generando infraestructura LATAM (Policías y Baches) en {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<additional>\n')
        
        # Puntos de Control Policial
        # Se asume un control en las avenidas principales entrantes al cruce B1 (edge B0B1 o similar)
        # Para simplificar, añadimos detectores de área con parámetro hasPolice
        police_edges = ["A0B0", "B2B1", "C1B1", "D1C1"]
        for edge in police_edges:
            f.write(f'  <laneAreaDetector id="police_{edge}" lane="{edge}_0" pos="0" length="50" freq="3600" file="police_out.xml">\n')
            f.write(f'    <param key="hasPolice" value="true"/>\n')
            f.write(f'  </laneAreaDetector>\n')
            
        # Baches (POIs) en ubicaciones aleatorias
        bache_edges = ["A1B1", "B1C1", "C2C1", "D2C2"]
        for i, edge in enumerate(bache_edges):
            f.write(f'  <poi id="bache_{i}" color="0,0,0" layer="3" lane="{edge}_0" pos="25">\n')
            f.write(f'    <param key="isBache" value="true"/>\n')
            f.write(f'  </poi>\n')
            
        f.write('</additional>\n')


def export_latam_scenarios(output_path: str, driver_stats: dict, include_pedestrians: bool = True):
    logger.info(f"Exportando escenarios LATAM a {output_path}...")
    
    sorted_drivers = sorted(
        driver_stats.items(),
        key=lambda x: x[1]['imprudence_score'],
        reverse=True
    )
    
    routes = [
        ("r_WE_0", "A0B0 B0C0 C0D0"), ("r_WE_1", "A1B1 B1C1 C1D1"), ("r_WE_2", "A2B2 B2C2 C2D2"), ("r_WE_3", "A3B3 B3C3 C3D3"),
        ("r_EW_0", "D0C0 C0B0 B0A0"), ("r_EW_1", "D1C1 C1B1 B1A1"), ("r_EW_2", "D2C2 C2B2 B2A2"), ("r_EW_3", "D3C3 C3B3 B3A3"),
        ("r_SN_A", "A0A1 A1A2 A2A3"), ("r_SN_B", "B0B1 B1B2 B2B3"), ("r_SN_C", "C0C1 C1C2 C2C3"), ("r_SN_D", "D0D1 D1D2 D2D3"),
        ("r_NS_A", "A3A2 A2A1 A1A0"), ("r_NS_B", "B3B2 B2B1 B1B0"), ("r_NS_C", "C3C2 C2C1 C1C0"), ("r_NS_D", "D3D2 D2D1 D1D0")
    ]
    edges = set()
    for _, edge_str in routes:
        for e in edge_str.split():
            edges.add(e)
    edges = list(edges)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        f.write('  <!-- Vehículos Especiales LATAM -->\n')
        # Motocicletas imprudentes (Lane Splitting)
        f.write('  <vType id="moto_imprudent" vClass="motorcycle" width="0.8" minGap="0.0" laneChangeModel="SL2015" lcSublane="1" tau="0.4" sigma="0.9" speedFactor="1.5" jmDriveAfterRedTime="3.0" lcCooperative="0.0" lcPushy="1.5" color="0,1,0"/>\n')
        # Micros / Combis
        f.write('  <vType id="micro_imprudent" vClass="bus" length="7.0" width="2.2" minGap="1.0" laneChangeModel="SL2015" tau="1.0" sigma="0.7" speedFactor="1.2" lcAssertive="1.2" lcImpatience="1.0" color="1,0.5,0">\n')
        f.write('    <param key="ocupacion" value="0"/>\n') # Custom param para TraCI
        f.write('  </vType>\n\n')
        
        if include_pedestrians:
            f.write('  <vType id="ped_imprudent" vClass="pedestrian" color="0,1,1" guiShape="pedestrian"/>\n\n')

        f.write('  <!-- Conductores de Automóviles -->\n')
        vtype_ids = []
        for idx, (driver_id, stats_data) in enumerate(sorted_drivers):
            vtype_id = f"imprudent_{driver_id}"
            vtype_ids.append(vtype_id)
            imp = stats_data['imprudence_score']
            
            sigma = max(0.1, min(1.0, 0.3 + imp * 1.5))
            tau = max(0.5, min(1.5, 1.5 - imp))
            speed_factor = max(1.0, min(1.5, 1.0 + imp * 0.4))
            accel = max(0.5, min(3.0, stats_data['avg_accel'] * 2.0))
            decel = max(0.75, min(4.5, accel * 1.5))
            
            # Intersecciones
            jmIgnoreKeepClearTime = "0" if imp > 0.3 else "5"
            jmIgnoreFoeProb = f"{min(0.4, imp * 0.8):.2f}"
            jmIgnorePriorityProb = f"{min(0.8, imp):.2f}"
            jmForcePriorityProb = f"{min(0.5, imp * 0.6):.2f}"
            jmDriveAfterYellowTime = f"{min(4.0, imp * 5.0):.1f}"
            jmDriveAfterRedTime = f"{min(2.0, imp * 3.0):.1f}"
            
            # Carril
            lcStrategic = f"{max(0.1, 1.0 - imp):.1f}"
            lcCooperative = f"{max(0.0, 1.0 - imp * 1.5):.1f}"
            lcSpeedGain = f"{min(3.0, 1.0 + imp * 3.0):.1f}"
            lcKeepRight = "0" if imp > 0.4 else "1"
            lcAssertive = f"{min(5.0, 1.0 + imp * 5.0):.1f}"
            lcPushy = "1" if imp > 0.6 else "0"
            
            color_val = max(0.0, 1.0 - imp)
            color_str = f"1,{color_val:.2f},{color_val:.2f}"
            
            f.write(f'  <vType id="{vtype_id}" sigma="{sigma:.3f}" tau="{tau:.3f}" ')
            f.write(f'speedFactor="{speed_factor:.3f}" accel="{accel:.3f}" decel="{decel:.3f}" color="{color_str}" ')
            f.write(f'jmIgnoreKeepClearTime="{jmIgnoreKeepClearTime}" ')
            f.write(f'jmIgnoreFoeProb="{jmIgnoreFoeProb}" ')
            f.write(f'jmDriveAfterYellowTime="{jmDriveAfterYellowTime}" jmDriveAfterRedTime="{jmDriveAfterRedTime}" ')
            f.write(f'laneChangeModel="SL2015" lcStrategic="{lcStrategic}" lcCooperative="{lcCooperative}" lcSpeedGain="{lcSpeedGain}" ')
            f.write(f'lcKeepRight="{lcKeepRight}" lcAssertive="{lcAssertive}" lcPushy="{lcPushy}">\n')
            # Parámetros extra para TraCI Chaos Manager
            f.write(f'    <param key="imprudence" value="{imp:.3f}"/>\n')
            f.write(f'    <param key="jmIgnorePriorityProb" value="{jmIgnorePriorityProb}"/>\n')
            f.write(f'    <param key="jmForcePriorityProb" value="{jmForcePriorityProb}"/>\n')
            f.write(f'  </vType>\n')

        f.write('\n  <!-- Rutas -->\n')
        for r_id, r_edges in routes:
            f.write(f'  <route id="{r_id}" edges="{r_edges}"/>\n')

        f.write('\n  <!-- Flujos de Tráfico Mixto -->\n')
        # Proporciones: 60% Autos (nuestros perfiles), 25% Motos, 15% Micros
        all_types = vtype_ids + ["moto_imprudent", "micro_imprudent"]
        prob_auto_total = 0.60
        prob_per_auto = prob_auto_total / len(vtype_ids)
        
        all_probs = [f"{prob_per_auto:.4f}" for _ in vtype_ids] + ["0.25", "0.15"]
        
        f.write(f'  <vTypeDistribution id="mixed_latam" vTypes="{" ".join(all_types)}" probabilities="{" ".join(all_probs)}"/>\n\n')

        for i, (r_id, _) in enumerate(routes):
            f.write(f'  <flow id="f_latam_{i:03d}" route="{r_id}" type="mixed_latam" begin="0" end="3600" period="10"/>\n')

        # Paradas indebidas estáticas (legacy, las dinámicas van por TraCI)
        # Seguiremos inyectando algunos autos estacionados
        net_path = os.path.join(os.path.dirname(output_path), "..", "..", "sumo_configs", "hangzhou", "hangzhou_pedestrian.net.xml")
        import xml.etree.ElementTree as ET
        valid_lanes = {}
        if os.path.exists(net_path):
            tree = ET.parse(net_path)
            for edge in tree.getroot().findall('edge'):
                if not edge.get('function'):
                    edge_id = edge.get('id')
                    for lane in edge.findall('lane'):
                        allow = lane.get('allow', '')
                        disallow = lane.get('disallow', '')
                        if 'pedestrian' in allow and len(allow.split()) == 1:
                            continue
                        if 'passenger' not in disallow:
                            valid_lanes[edge_id] = lane.get('id')
                            break
                            
        for i in range(25):
            vtype = random.choice(vtype_ids)
            r_id, r_edges = random.choice(routes)
            edges_list = r_edges.split()
            stop_edge = random.choice(edges_list)
            stop_lane = valid_lanes.get(stop_edge, f"{stop_edge}_1")
            stop_duration = random.randint(60, 300)
            depart_time = random.randint(0, 60)
            
            f.write(f'  <vehicle id="parked_{i:03d}" type="{vtype}" route="{r_id}" depart="{depart_time}" color="1,0,0">\n')
            f.write(f'    <stop lane="{stop_lane}" duration="{stop_duration}"/>\n')
            f.write(f'  </vehicle>\n')

        if include_pedestrians:
            f.write('\n  <!-- Peatones -->\n')
            ped_edges = list(valid_lanes.keys()) if valid_lanes else [e for _, e_str in routes for e in e_str.split()]
            for i in range(20):
                from_edge = random.choice(ped_edges)
                to_edge = random.choice(ped_edges)
                while from_edge == to_edge:
                    to_edge = random.choice(ped_edges)
                
                # Probability 0.3 de cruzar fuera de paso de cebra o generar arrastre
                f.write(f'  <personFlow id="ped_flow_{i:03d}" type="ped_imprudent" begin="0" end="3600" period="{random.randint(5, 15)}">\n')
                f.write(f'    <param key="jaywalk_prob" value="0.3"/>\n')
                f.write(f'    <walk from="{from_edge}" to="{to_edge}"/>\n')
                f.write(f'  </personFlow>\n')

        f.write('\n  <!-- Peatones -->\n')
        ped_edges = list(valid_lanes.keys()) if valid_lanes else [e for _, e_str in routes for e in e_str.split()]
        if ped_edges:
            for i in range(300):
                e_from = random.choice(ped_edges)
                e_to = random.choice(ped_edges)
                depart = random.randint(0, 720)
                f.write(f'  <person id="ped_{i}" depart="{depart}" type="ped_imprudent">\n')
                f.write(f'    <walk from="{e_from}" to="{e_to}"/>\n')
                f.write(f'  </person>\n')

        f.write('</routes>\n')
    logger.info(f"Archivo exportado: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='../experiments/hangzhou_robustness/scenarios')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-pedestrians', action='store_true')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    df = load_data(args.data_path)
    driver_stats = analyze_drivers(df)
    
    analysis_path = output_dir / 'latam_driver_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(driver_stats, f, indent=2)
        
    export_latam_infrastructure(str(output_dir / 'latam_infrastructure.add.xml'))
    sumo_output = output_dir / 'latam_imprudent_drivers.rou.xml'
    export_latam_scenarios(str(sumo_output), driver_stats, not args.no_pedestrians)
    logger.info("GENERACIÓN LATAM V2 COMPLETADA")

if __name__ == '__main__':
    main()
