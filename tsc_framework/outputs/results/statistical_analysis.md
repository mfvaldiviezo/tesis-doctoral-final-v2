## [ANALISIS] Análisis Estadístico — Pruebas No-Paramétricas

> **Nota metodológica:** Dado que la distribución de los retrasos vehiculares no es normal (presencia de episodios catastróficos de gridlock), se aplican pruebas no-paramétricas robustas ante outliers: Mann-Whitney U (comparación por pares) y Kruskal-Wallis (comparación multi-grupo). El nivel de significancia es α = 0.05. El tamaño del efecto se mide mediante la correlación rank-biserial *r* (|r| < 0.3 = pequeño; 0.3–0.5 = medio; > 0.5 = grande).

---

### Comparación H-SARG Caótico vs H-SARG Ideal (Mann-Whitney U)

| Caos % | U stat | p-value | Signif. | r (effect size) | Magnitud | Mediana H-SARG Caos | Mediana H-SARG Ideal |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0% | 51.5 | 0.9397 | ns | -0.030 | negligible | 566.3 s | 560.3 s |
| 15% | 28.0 | 0.1041 | ns | 0.440 | medium | 617.4 s | 882.5 s |
| 30% | 46.0 | 0.7913 | ns | 0.080 | negligible | 594.9 s | 683.9 s |
| 50% | 48.0 | 0.9097 | ns | 0.040 | negligible | 622.4 s | 671.6 s |

**Leyenda:** *** p < 0.001 | ** p < 0.01 | * p < 0.05 | ns = no significativo

---

### Diferencias entre Todos los Controladores (Kruskal-Wallis)

| Caos % | H stat | p-value | Signif. | Interpretación |
|:---:|:---:|:---:|:---:|:---|
| 0% | 10.297 | 0.0357 | * | Diferencias significativas entre controladores |
| 15% | 26.785 | 0.0000 | *** | Diferencias significativas entre controladores |
| 30% | 23.937 | 0.0001 | *** | Diferencias significativas entre controladores |
| 50% | 16.237 | 0.0027 | ** | Diferencias significativas entre controladores |

---

### Tabla de Tasa de Colapso de Red (Gridlock Rate)

*(Umbral de gridlock: Delay > 2000 s por episodio)*

| Controlador | Caos % | Tasa Colapso (%) | Mediana (s) | IQR (s) | CVaR₉₅ (s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **H-SARG Caótico** | 0% | **0.0%** | 566.3 s | 161.0 s | 859.2 s |
| **H-SARG Caótico** | 15% | **0.0%** | 617.4 s | 409.6 s | 1322.9 s |
| **H-SARG Caótico** | 30% | **30.0%** | 594.9 s | 8527.8 s | 99555.2 s |
| **H-SARG Caótico** | 50% | **10.0%** | 622.4 s | 362.1 s | 163746.3 s |
| **H-SARG Ideal** | 0% | **0.0%** | 560.3 s | 78.1 s | 854.0 s |
| **H-SARG Ideal** | 15% | **10.0%** | 882.5 s | 358.7 s | 535029.5 s |
| **H-SARG Ideal** | 30% | **0.0%** | 683.9 s | 157.5 s | 922.1 s |
| **H-SARG Ideal** | 50% | **0.0%** | 671.6 s | 555.2 s | 1586.9 s |

---
*Análisis generado automáticamente por `statistical_analysis.py` — TSC Framework Doctoral.*
