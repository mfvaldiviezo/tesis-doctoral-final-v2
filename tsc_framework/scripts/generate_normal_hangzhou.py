#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path

def export_normal_scenarios(output_path: str):
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
        
        f.write('  <!-- Tráfico Normal (Baseline) -->\n')
        f.write('  <vType id="ped_normal" vClass="pedestrian" color="0,1,1" guiShape="pedestrian"/>\n')
        f.write('  <vType id="car" length="4.5" minGap="2.5" maxSpeed="13.89" sigma="0.5" tau="1.0" color="1,1,1" laneChangeModel="SL2015"/>\n')
        f.write('  <vType id="moto" vClass="motorcycle" width="0.8" minGap="1.0" maxSpeed="15.0" sigma="0.5" tau="1.0" color="0,1,0" laneChangeModel="SL2015"/>\n')
        f.write('  <vType id="bus" vClass="bus" length="7.0" width="2.2" minGap="2.5" maxSpeed="12.0" sigma="0.5" tau="1.0" color="1,0.5,0" laneChangeModel="SL2015"/>\n\n')
        f.write('  <vTypeDistribution id="mixed_normal" vTypes="car moto bus" probabilities="0.60 0.25 0.15"/>\n\n')
        
        f.write('\n  <!-- Rutas de la Red Hangzhou 4x4 -->\n')
        for r_id, r_edges in routes:
            f.write(f'  <route id="{r_id}" edges="{r_edges}"/>\n')

        f.write('\n  <!-- Flujos de Tráfico Normal -->\n')
        for i, (r_id, _) in enumerate(routes):
            # Mismo periodo que el tráfico mixto LATAM para igualar cantidad de vehículos
            f.write(f'  <flow id="f_normal_{i:03d}" route="{r_id}" type="mixed_normal" begin="0" end="3600" period="10"/>\n')

        # Agregamos los 50 vehículos adicionales pero circulando normal (sin paradas ni bloqueo)
        random.seed(42)
        f.write('\n  <!-- Vehículos extra (equivalentes a los estacionados en LATAM pero circulando normal) -->\n')
        for i in range(50):
            r_id, r_edges = random.choice(routes)
            depart_time = random.randint(0, 60)
            f.write(f'  <vehicle id="normal_{i:03d}" type="car" route="{r_id}" depart="{depart_time}"/>\n')

        f.write('\n  <!-- Peatones -->\n')
        for i in range(300):
            e_from = random.choice(edges)
            e_to = random.choice(edges)
            depart = random.randint(0, 720)
            f.write(f'  <person id="ped_{i}" depart="{depart}" type="ped_normal">\n')
            f.write(f'    <walk from="{e_from}" to="{e_to}"/>\n')
            f.write(f'  </person>\n')

        f.write('</routes>\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='../experiments/hangzhou_robustness/scenarios')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'hangzhou_normal_drivers.rou.xml'
    export_normal_scenarios(str(out_path))
    print(f"Normal drivers scenario generated at {out_path}")
