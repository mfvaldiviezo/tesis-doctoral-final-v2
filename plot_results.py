import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configurar estilo visual para publicación
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Importar la lógica de búsqueda de evaluate_marl
try:
    from evaluate_marl import ALGORITHMS, SCENARIOS, find_metrics
except ImportError:
    print("Error: No se pudo importar evaluate_marl.py. Asegúrate de ejecutar este script desde la raíz del proyecto.")
    sys.exit(1)

def gather_data():
    data = []
    for algo_tuple in ALGORITHMS:
        algo_id = algo_tuple[0]
        algo_name = algo_tuple[1]
        for scen_tuple in SCENARIOS:
            scen_id = scen_tuple[0]
            m = find_metrics(algo_id, scen_id)
            if m:
                # Filtrar campos faltantes o inválidos
                throughput = m.get("throughput_per_step")
                gini = m.get("gini_temporal")
                cvar = m.get("cvar95_queue")
                if all(isinstance(x, (int, float)) for x in [throughput, gini, cvar]):
                    data.append({
                        "Algorithm": algo_name,
                        "Scenario": "Ideal" if scen_id == "ideal" else "Caótico (LATAM)",
                        "Throughput": float(throughput),
                        "Gini": float(gini),
                        "CVaR95": float(cvar)
                    })
    return data

def plot_resilience_bar_chart(data, output_dir):
    """Genera un gráfico de barras doble mostrando el cambio en Throughput."""
    plt.figure(figsize=(10, 6))
    
    # Extraer datos para el plot
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Crear barplot
    ax = sns.barplot(x="Algorithm", y="Throughput", hue="Scenario", data=df, palette=["#3498db", "#e74c3c"])
    
    plt.title("Resiliencia de Throughput: Ideal vs Caótico (LATAM)", pad=20, weight='bold')
    plt.ylabel("Throughput (veh/step)", weight='bold')
    plt.xlabel("Algoritmo", weight='bold')
    plt.legend(title="Escenario de Tráfico")
    
    # Añadir valores sobre las barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    plt.tight_layout()
    out_path = output_dir / "resilience_throughput.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {out_path}")
    plt.close()

def plot_efficiency_vs_equity(data, output_dir):
    """Genera un gráfico de dispersión (scatter) Gini vs Throughput."""
    plt.figure(figsize=(10, 8))
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Plot por escenario con diferentes marcadores
    ideal_df = df[df["Scenario"] == "Ideal"]
    latam_df = df[df["Scenario"] == "Caótico (LATAM)"]
    
    colors = sns.color_palette("husl", len(df["Algorithm"].unique()))
    algo_color_map = dict(zip(df["Algorithm"].unique(), colors))

    # Scatter Ideal (Círculos)
    for _, row in ideal_df.iterrows():
        plt.scatter(row["Gini"], row["Throughput"], 
                    color=algo_color_map[row["Algorithm"]], 
                    marker='o', s=200, edgecolors='black', alpha=0.7)

    # Scatter Latam (Estrellas o Cruces rojas)
    for _, row in latam_df.iterrows():
        plt.scatter(row["Gini"], row["Throughput"], 
                    color=algo_color_map[row["Algorithm"]], 
                    marker='X', s=300, edgecolors='black')
        
        # Conectar ideal con latam con una flecha para mostrar la migración (degradación)
        ideal_row = ideal_df[ideal_df["Algorithm"] == row["Algorithm"]]
        if not ideal_row.empty:
            ix, iy = ideal_row.iloc[0]["Gini"], ideal_row.iloc[0]["Throughput"]
            nx, ny = row["Gini"], row["Throughput"]
            plt.annotate("", xy=(nx, ny), xytext=(ix, iy),
                         arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
            
            # Etiqueta del algoritmo en el punto LATAM
            plt.text(nx + 0.01, ny, row["Algorithm"], fontsize=12, weight='bold')

    plt.title("Trade-off Eficiencia vs Equidad\n(Flechas indican degradación hacia el Caos LATAM)", pad=20, weight='bold')
    
    # Ejes invertidos lógicamente: Mejor Gini es MENOR (hacia la izquierda)
    plt.xlabel("Injusticia Espacial (Gini Temporal) → Peor", weight='bold')
    plt.ylabel("Eficiencia Global (Throughput veh/step) → Mejor", weight='bold')
    
    # Leyenda customizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Tráfico Ideal', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='Tráfico Caótico', markerfacecolor='gray', markersize=12),
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    out_path = output_dir / "efficiency_vs_equity.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {out_path}")
    plt.close()

if __name__ == "__main__":
    print("Recopilando datos para gráficas...")
    data = gather_data()
    
    if not data:
        print("❌ No se encontraron datos válidos. Asegúrate de haber ejecutado evaluate_marl.py primero.")
        sys.exit(1)
        
    output_dir = Path("benchmark_reports/plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_resilience_bar_chart(data, output_dir)
    plot_efficiency_vs_equity(data, output_dir)
    
    print("\n🎉 ¡Gráficas de nivel publicación generadas con éxito!")
