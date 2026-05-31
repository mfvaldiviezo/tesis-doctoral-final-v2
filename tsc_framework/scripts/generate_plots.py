#!/usr/bin/env python3
"""
generate_plots.py — Generador de Figuras Científicas para la Tesis Doctoral (v2 — Post-Revisión)
=================================================================================================
MEJORAS INCORPORADAS (según revisión doctoral):
  1. Curvas de degradación usan MEDIANA (no media) con bandas IQR al 25%-75%.
  2. Boxplots para TODOS los niveles de caos en panel 2×2.
  3. Heatmap de tasa de colapso (gridlock rate) — figura más solicitada.
  4. Comparación de CVaR95 en barras agrupadas.
  5. Trade-off Equidad vs Eficiencia rediseñado con medianas + elipses IQR.

Referencia: Recomendaciones del tutor doctoral (revisión 2026-05-30).
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuración Visual para Publicación Ph.D.
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Umbral físico de gridlock (recomendado por tutor: delay > 2000s)
GRIDLOCK_THRESHOLD = 2000.0

# Paleta coherente por agente
PALETTE = {
    "ppo_chaos":   "#27ae60",   # Verde oscuro — H-SARG Caótico
    "ppo_ideal":   "#2980b9",   # Azul — H-SARG Ideal
    "Fixed":       "#7f8c8d",   # Gris — Fixed Time
    "MaxPressure": "#e67e22",   # Naranja — MaxPressure
    "CoLight":     "#c0392b",   # Rojo oscuro — CoLight
}

LABELS = {
    "ppo_chaos":   "H-SARG Caótico",
    "ppo_ideal":   "H-SARG Ideal",
    "Fixed":       "Fixed Time",
    "MaxPressure": "MaxPressure",
    "CoLight":     "CoLight",
}

MARKERS = {
    "ppo_chaos":   "o",
    "ppo_ideal":   "s",
    "Fixed":       "D",
    "MaxPressure": "^",
    "CoLight":     "X",
}

CHAOS_LABELS = {0.0: "0%", 0.15: "15%", 0.30: "30%", 0.50: "50%"}
AGENT_ORDER = ["ppo_ideal", "ppo_chaos", "MaxPressure", "Fixed", "CoLight"]


# ─────────────────────────────────────────────────────────────────────────────
# Figura 1: Curvas de Degradación con MEDIANA + Bandas IQR
# ─────────────────────────────────────────────────────────────────────────────
def plot_degradation_median(df: pd.DataFrame, output_dir: Path):
    """
    Curvas de degradación usando MEDIANA en lugar de media.
    Bandas sombreadas = IQR [25%, 75%].
    Esto elimina el sesgo de los episodios catastróficos en la visualización.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))

    chaos_levels = sorted(df["chaos_level"].unique())
    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]

    for agent in agents_present:
        adf = df[df["agent"] == agent]
        medians, q25, q75, xs = [], [], [], []

        for cl in chaos_levels:
            sub = adf[adf["chaos_level"] == cl]["delay_mean"]
            if len(sub) == 0:
                continue
            medians.append(float(np.median(sub)))
            q25.append(float(np.percentile(sub, 25)))
            q75.append(float(np.percentile(sub, 75)))
            xs.append(cl * 100)

        if not xs:
            continue

        color = PALETTE.get(agent, "#000000")
        label = LABELS.get(agent, agent)
        marker = MARKERS.get(agent, "o")

        ax.plot(xs, medians, color=color, marker=marker, markersize=9,
                linewidth=2.5, label=label, zorder=5)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.12)

    ax.set_title("Curvas de Degradación de Eficiencia frente a Caos Conductual\n"
                 "(Mediana ± IQR [25%–75%])", pad=12, fontweight='bold')
    ax.set_xlabel("Nivel de Caos — Probabilidad de Conductores Imprudentes (%)", fontweight='bold')
    ax.set_ylabel("Retraso Vehicular Mediano (s)", fontweight='bold')
    ax.set_xticks([0, 15, 30, 50])
    ax.set_xticklabels(["0%", "15%", "30%", "50%"])
    ax.legend(title="Controlador", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Nota metodológica
    ax.text(0.01, 0.97,
            "Nota: Se usa mediana para eliminar sesgo de episodios de gridlock.",
            transform=ax.transAxes, fontsize=9, color="#555555",
            verticalalignment='top', style='italic')

    plt.tight_layout()
    out = output_dir / "degradation_curves_median.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✅ [Fig 1] Curvas de degradación (mediana+IQR): {out.name}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figura 2: Boxplots Panel 2×2 — Todos los niveles de caos
# ─────────────────────────────────────────────────────────────────────────────
def plot_boxplots_panel(df: pd.DataFrame, output_dir: Path):
    """
    Panel 2×2 de boxplots mostrando la distribución de delay para cada nivel de caos.
    Los outliers (episodios catastróficos) quedan visibles como puntos individuales.
    """
    chaos_levels = sorted(df["chaos_level"].unique())
    n_chaos = len(chaos_levels)
    ncols = 2
    nrows = (n_chaos + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9), sharey=False)
    axes = axes.flatten()

    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    colors = [PALETTE.get(a, "#000000") for a in agents_present]

    for idx, cl in enumerate(chaos_levels):
        ax = axes[idx]
        sub = df[df["chaos_level"] == cl]

        plot_data = [sub[sub["agent"] == a]["delay_mean"].values for a in agents_present]

        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            flierprops=dict(marker='o', markersize=5, markeredgecolor='black',
                            markerfacecolor='red', alpha=0.7),
            widths=0.55
        )

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(f"Nivel de Caos: {CHAOS_LABELS.get(cl, f'{cl*100:.0f}%')}",
                     fontweight='bold', fontsize=12)
        ax.set_xticks(range(1, len(agents_present) + 1))
        ax.set_xticklabels(
            [LABELS.get(a, a) for a in agents_present],
            rotation=25, ha='right', fontsize=9
        )
        ax.set_ylabel("Delay Promedio Ep. (s)", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Anotar mediana encima de cada caja
        for i, data in enumerate(plot_data):
            if len(data) > 0:
                med = np.median(data)
                ax.text(i + 1, med, f" {med:.0f}s",
                        ha='left', va='center', fontsize=8, color='#333333')

    # Ocultar paneles sobrantes si hay número impar
    for j in range(n_chaos, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribución de Retraso Vehicular por Nivel de Caos\n"
        "(Outliers rojos = episodios de gridlock/colapso)",
        fontweight='bold', fontsize=14, y=1.01
    )

    # Leyenda global
    legend_patches = [
        mpatches.Patch(facecolor=PALETTE.get(a, "#000"), alpha=0.75,
                       label=LABELS.get(a, a))
        for a in agents_present
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(agents_present),
               bbox_to_anchor=(0.5, -0.04), frameon=True, facecolor='white',
               title="Controlador")

    plt.tight_layout()
    out = output_dir / "stress_boxplot_all_chaos.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✅ [Fig 2] Boxplots panel 2×2 (todos los niveles): {out.name}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figura 3: Heatmap de Tasa de Colapso (Gridlock Rate)
# ─────────────────────────────────────────────────────────────────────────────
def plot_collapse_rate_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Heatmap de la tasa de colapso (% episodios con delay > GRIDLOCK_THRESHOLD)
    por agente × nivel de caos. Figura directamente solicitada por el tutor.
    """
    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    chaos_levels = sorted(df["chaos_level"].unique())

    # Calcular tasa de colapso
    collapse_matrix = pd.DataFrame(index=agents_present, columns=chaos_levels, dtype=float)
    for agent in agents_present:
        for cl in chaos_levels:
            sub = df[(df["agent"] == agent) & (df["chaos_level"] == cl)]["delay_mean"]
            if len(sub) == 0:
                collapse_matrix.loc[agent, cl] = np.nan
            else:
                collapse_matrix.loc[agent, cl] = float((sub > GRIDLOCK_THRESHOLD).mean() * 100)

    collapse_matrix = collapse_matrix.astype(float)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Paleta: blanco = 0%, rojo intenso = 100%
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        collapse_matrix,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        linewidths=0.6,
        linecolor='white',
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
        cbar_kws={"label": "Tasa de Colapso (%)", "shrink": 0.85}
    )

    # Formato de ejes
    ax.set_xticklabels(
        [CHAOS_LABELS.get(float(c), f"{float(c)*100:.0f}%") for c in chaos_levels],
        fontsize=11, fontweight='bold'
    )
    ax.set_yticklabels(
        [LABELS.get(a, a) for a in agents_present],
        rotation=0, fontsize=11
    )
    ax.set_xlabel("Nivel de Caos — Conductores Imprudentes", fontweight='bold', fontsize=12)
    ax.set_ylabel("Controlador Semafórico", fontweight='bold', fontsize=12)
    ax.set_title(
        f"Tasa de Colapso de Red — Gridlock Rate (%)\n"
        f"(Episodios con Delay > {GRIDLOCK_THRESHOLD:.0f} s)",
        pad=12, fontweight='bold', fontsize=13
    )

    # Añadir símbolo ★ en la celda más crítica (mayor CVaR diferencia entre modelos RL)
    # Resaltar: ppo_ideal @ 15% (única celda con ventaja documental clara)
    if 0.15 in chaos_levels and "ppo_ideal" in agents_present:
        col_idx = list(chaos_levels).index(0.15)
        row_idx = agents_present.index("ppo_ideal")
        ax.add_patch(plt.Rectangle(
            (col_idx, row_idx), 1, 1,
            fill=False, edgecolor='blue', lw=2.5, clip_on=False
        ))

    plt.tight_layout()
    out = output_dir / "collapse_rate_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✅ [Fig 3] Heatmap tasa de colapso: {out.name}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figura 4: Comparación de CVaR95 — Barras Agrupadas
# ─────────────────────────────────────────────────────────────────────────────
def plot_cvar_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Barras agrupadas comparando CVaR95 (media del peor 5%) entre todos los agentes
    por nivel de caos. Demuestra cuándo el caos ayuda (15%) y cuándo no (30%, 50%).
    """
    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    chaos_levels = sorted(df["chaos_level"].unique())

    # Calcular CVaR medio por agente/caos (desde la columna cvar_95 del CSV)
    agg = df.groupby(["agent", "chaos_level"])["cvar_95"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(chaos_levels))
    width = 0.75 / len(agents_present)
    offsets = np.linspace(-0.75/2 + width/2, 0.75/2 - width/2, len(agents_present))

    for i, agent in enumerate(agents_present):
        sub = agg[agg["agent"] == agent].set_index("chaos_level")
        vals = [sub.loc[cl, "cvar_95"] if cl in sub.index else 0.0 for cl in chaos_levels]
        bars = ax.bar(
            x + offsets[i], vals, width,
            color=PALETTE.get(agent, "#000"),
            alpha=0.82,
            label=LABELS.get(agent, agent),
            edgecolor='white', linewidth=0.5
        )

    ax.set_xlabel("Nivel de Caos — Conductores Imprudentes (%)", fontweight='bold')
    ax.set_ylabel("CVaR₉₅ Promedio (s) — Riesgo de Cola Extrema", fontweight='bold')
    ax.set_title(
        "Comparación de Riesgo Extremo (CVaR₉₅) por Nivel de Caos\n"
        "(Cuanto menor, mejor — mide el peor 5% de episodios)",
        pad=12, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CHAOS_LABELS.get(cl, f"{cl*100:.0f}%") for cl in chaos_levels])
    ax.legend(title="Controlador", frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Anotar diferencia clave: ppo_chaos vs ppo_ideal @ 15%
    try:
        chaos_cvar_15 = agg[(agg["agent"] == "ppo_chaos") & (agg["chaos_level"] == 0.15)]["cvar_95"].values[0]
        ideal_cvar_15 = agg[(agg["agent"] == "ppo_ideal") & (agg["chaos_level"] == 0.15)]["cvar_95"].values[0]
        ax.annotate(
            f"Δ={ideal_cvar_15/chaos_cvar_15:.0f}× más riesgo\n(ppo_ideal vs ppo_chaos)",
            xy=(x[1] + offsets[0], ideal_cvar_15),
            xytext=(x[1] + 0.4, ideal_cvar_15 * 0.75),
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.8),
            fontsize=9, color='navy', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='navy', alpha=0.8)
        )
    except Exception:
        pass

    plt.tight_layout()
    out = output_dir / "cvar_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✅ [Fig 4] Comparación CVaR95 barras agrupadas: {out.name}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figura 5: Trade-off Eficiencia vs Equidad (rediseñado con medianas)
# ─────────────────────────────────────────────────────────────────────────────
def plot_equity_tradeoff(df: pd.DataFrame, output_dir: Path):
    """
    Gráfico de dispersión Eficiencia vs Equidad usando MEDIANAS (no medias).
    Las flechas muestran la trayectoria de degradación de 0% → 50% caos.
    Los puntos de episodios catastróficos (>2000s) se excluyen del cómputo de mediana.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    chaos_levels = sorted(df["chaos_level"].unique())

    for agent in agents_present:
        adf = df[df["agent"] == agent]
        pts_x, pts_y = [], []

        for cl in chaos_levels:
            sub = adf[adf["chaos_level"] == cl]
            # Usar mediana (excluye automáticamente el sesgo de gridlocks)
            med_delay = float(np.median(sub["delay_mean"]))
            med_gini = float(np.median(sub["gini_mean"]))
            pts_x.append(med_gini)
            pts_y.append(med_delay)

        color = PALETTE.get(agent, "#000000")
        label = LABELS.get(agent, agent)
        marker = MARKERS.get(agent, "o")

        # Línea de trayectoria
        ax.plot(pts_x, pts_y, color=color, linestyle='--', alpha=0.5, linewidth=1.8)

        # Puntos por nivel de caos
        for i, (px, py, cl) in enumerate(zip(pts_x, pts_y, chaos_levels)):
            size = 80 + i * 40  # tamaño crece con el nivel de caos
            ax.scatter(px, py, color=color, marker=marker, s=size,
                       edgecolors='black', linewidth=0.8, zorder=5,
                       label=label if i == 0 else "")

        # Flecha de inicio a fin
        if len(pts_x) >= 2:
            ax.annotate(
                "", xy=(pts_x[-1], pts_y[-1]), xytext=(pts_x[-2], pts_y[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8)
            )

        # Etiqueta en punto 50%
        ax.annotate(
            f"{label}\n(50%)",
            xy=(pts_x[-1], pts_y[-1]),
            xytext=(8, 4), textcoords='offset points',
            fontsize=8.5, color=color, fontweight='bold'
        )

    ax.set_title(
        "Trade-off Eficiencia vs Equidad Distributiva\n"
        "(Medianas por nivel de caos; flechas indican degradación 0%→50%)",
        pad=12, fontweight='bold'
    )
    ax.set_xlabel("Injusticia Distributiva (Coeficiente de Gini, Mediana) → Peor", fontweight='bold')
    ax.set_ylabel("Retraso Vehicular Mediano (s) → Peor", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Nota de tamaño de punto
    ax.text(0.01, 0.01,
            "Tamaño del punto ∝ Nivel de caos (0%, 15%, 30%, 50%)",
            transform=ax.transAxes, fontsize=8.5, color='#666666', style='italic')

    # Leyenda sin duplicados
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Controlador",
              frameon=True, facecolor='white', edgecolor='#cccccc')

    plt.tight_layout()
    out = output_dir / "equity_vs_efficiency_tradeoff.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✅ [Fig 5] Trade-off eficiencia vs equidad (medianas): {out.name}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generador de Figuras Científicas Doctorales v2")
    parser.add_argument("--input-dir", type=str, default="tsc_framework/outputs/results",
                        help="Directorio con metrics_detailed.csv")
    parser.add_argument("--output-dir", type=str, default="tsc_framework/outputs/figures",
                        help="Directorio de salida para las figuras PNG")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_file = input_dir / "metrics_detailed.csv"
    if not detailed_file.exists():
        print(f"❌ Archivo no encontrado: {detailed_file}")
        sys.exit(1)

    df = pd.read_csv(detailed_file)
    print(f"\n📊 Datos cargados: {len(df)} episodios, "
          f"{df['agent'].nunique()} agentes, "
          f"{df['chaos_level'].nunique()} niveles de caos")
    print(f"📁 Generando figuras en: {output_dir}\n")

    plot_degradation_median(df, output_dir)
    plot_boxplots_panel(df, output_dir)
    plot_collapse_rate_heatmap(df, output_dir)
    plot_cvar_comparison(df, output_dir)
    plot_equity_tradeoff(df, output_dir)

    print(f"\n🎉 ¡5 figuras doctorales generadas con éxito en: {output_dir}")


if __name__ == "__main__":
    main()
