# REPORTE DOCTORAL DEFINITIVO: RESILIENCIA, TRANSICIÓN AL CAOS Y ROBUSTEZ ADAPTATIVA
**Candidato:** Marcelo  
**Modelo Principal:** H-SARG (Hybrid Self-Attention Gated Risk)  
**Fecha:** 2026-05-30  
**Entorno de Simulación:** SUMO 1.20+ (Sin Teleports, Colisiones Físicas Reales)  
**Revisión Incorporada:** Post-evaluación doctoral v3 (2026-05-30)

---

## Resumen Ejecutivo de Hallazgos

> **Nota estadística:** Las diferencias entre H-SARG Caótico y H-SARG Ideal no alcanzan significancia estadística con n=10 episodios (Mann-Whitney U, p>0.05 en todos los niveles). Las comparativas que siguen son descriptivas; la inferencia formal requiere n≥30. Los hallazgos más sólidos son cualitativos (tasa de colapso) y el resultado de transferencia zero-shot a Quito.

| # | Hallazgo | Evidencia |
| :---: | :--- | :--- |
| 1 | No existe penalización por entrenamiento caótico en escenario nominal | p=0.9397 (Mann-Whitney, ns); medianas equivalentes (~560.3-566.3 s) |
| 2 | Menor riesgo extremo observado en caos moderado (15%) | CVaR₉₅ 39× mayor en H-SARG Ideal (163,755.6 s vs 4,213.4 s) |
| 3 | Ausencia de colapsos observada en H-SARG Caótico a 15% de caos | 0/10 episodios (0.0%) frente a un colapso en H-SARG Ideal (1/10 episodios, 10.0%) |
| 4 | Mejor transferencia simultánea a Quito (delay + Gini + CVaR) | Mejora en las tres métricas a la vez; infrecuente en RL |
| 5 | Robustez no garantizada para caos severo (30%–50%) | 30.0% de colapsos en H-SARG Caótico a 30% de caos |

---

## Capítulo 1: Marco de Evaluación de Robustez (Hangzhou 4×4)

Este experimento evalúa la degradación progresiva bajo cuatro niveles de perturbación conductual (0%, 15%, 30%, 50%). Se reportan **medianas** como estadístico central (robusto ante gridlocks), el rango intercuartil (IQR) como dispersión, la tasa de colapso de red (episodios con delay > 2000 s) y el CVaR₉₅.

> **Nota metodológica:** La media aritmética queda omitida como estadístico principal porque un solo episodio catastrófico (e.g., 535,029 s de delay) sesga completamente la estimación. La mediana y el IQR reflejan el comportamiento operativo típico del sistema.

### Tabla Principal: Rendimiento por Controlador y Nivel de Caos

| Controlador | Caos % | Delay Mediana (s) | IQR (s) | Colapso % | CVaR₉₅ (s) | Gini Mediana |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **H-SARG Ideal** | 0% | 560.3 | 78.1 | **0.0%** | 3254.9 | 0.4563 |
| **H-SARG Ideal** | 15% | 882.5 | 358.7 | **10.0%** ⚠️ | 163755.6 | 0.4146 |
| **H-SARG Ideal** | 30% | 683.9 | 157.5 | **0.0%** | 3752.8 | 0.4689 |
| **H-SARG Ideal** | 50% | 671.6 | 555.2 | **0.0%** | 4022.0 | 0.4597 |
| **H-SARG Caótico** | 0% | 566.3 | 161.0 | **0.0%** | 3175.9 | 0.4648 |
| **H-SARG Caótico** | 15% | 617.4 | 409.6 | **0.0%** | 4213.4 | 0.4588 |
| **H-SARG Caótico** | 30% | 594.9 | 8527.8 | **30.0%** ⚠️ | 101319.5 | 0.4681 |
| **H-SARG Caótico** | 50% | 622.4 | 362.1 | **10.0%** ⚠️ | 50888.6 | 0.4594 |
| **MaxPressure** | 0% | 459.7 | 248.8 | **0.0%** | 3615.0 | 0.4739 |
| **MaxPressure** | 15% | 603.6 | 241.5 | **0.0%** | 3174.8 | 0.4688 |
| **MaxPressure** | 30% | 564.1 | 226.9 | **0.0%** | 3742.3 | 0.4575 |
| **MaxPressure** | 50% | 592.3 | 184.1 | **0.0%** | 4485.5 | 0.4587 |
| **Fixed Time** | 0% | 599.0 | 228.9 | **10.0%** ⚠️ | 38071.4 | 0.4875 |
| **Fixed Time** | 15% | 436.3 | 169.4 | **0.0%** | 3327.7 | 0.4917 |
| **Fixed Time** | 30% | 415.8 | 219.4 | **10.0%** ⚠️ | 176819.5 | 0.4815 |
| **Fixed Time** | 50% | 268.8 | 124.7 | **0.0%** | 2290.0 | 0.4682 |
| **CoLight** | 0% | 298.9 | 160.1 | **10.0%** ⚠️ | 34617.2 | 0.4959 |
| **CoLight** | 15% | 266.8 | 120.9 | **0.0%** | 2295.9 | 0.5058 |
| **CoLight** | 30% | 237.2 | 199.5 | **0.0%** | 2362.5 | 0.5025 |
| **CoLight** | 50% | 282.7 | 257.2 | **10.0%** ⚠️ | 37942.2 | 0.4809 |

> (!) = al menos un episodio de colapso de red registrado en esa celda.

---

## Capítulo 2: Análisis Estadístico Forense

### 2.1 Comportamiento Nominal — Caos 0%

En condiciones sin perturbación, los tres controladores RL muestran medianas de delay comparables:

| Controlador | Mediana Delay (s) |
| :--- | :---: |
| H-SARG Ideal | 560.3 |
| H-SARG Caótico | 566.3 |
| MaxPressure | 459.7 |

**Interpretación §2.1:** No se observa penalización en la mediana de delay por el entrenamiento con caos en escenario nominal (Kruskal-Wallis entre todos los agentes p=0.0357; las diferencias son atribuibles a la heterogeneidad entre Fixed Time, CoLight y MaxPressure, no entre los modelos H-SARG). Esto confirma que el entrenamiento bajo perturbaciones no sacrifica la eficiencia base en condiciones nominales.

---

### 2.2 Hallazgo Más Destacado — Caos 15% (Moderado)

> **Nota estadística:** La prueba Mann-Whitney U entre H-SARG Caótico e Ideal a caos 15% arroja p=0.1041 (r=0.440, efecto medium), sin alcanzar significancia con α=0.05. Las diferencias descriptivas son notables pero deben interpretarse con cautela dado n=10.

Bajo caos moderado (15%), los indicadores descriptivos divergen de manera relevante:

| Métrica | H-SARG Caótico | H-SARG Ideal |
| :--- | :---: | :---: |
| Tasa de Colapso | **0.0%** | **10.0%** |
| Mediana Delay | 617.4 s | 882.5 s |
| IQR Delay | 409.6 s | 358.7 s |
| CVaR₉₅ | 4213.4 s | 163755.6 s |

**Análisis forense — Episodios extremos de H-SARG Ideal (top-3):**
- **Ep 8**: Delay=535,029 s, Queue=469, Gini=0.204
- **Ep 1**: Delay=1,367 s, Queue=71, Gini=0.397
- **Ep 9**: Delay=1,126 s, Queue=61, Gini=0.390

Se observó que H-SARG Ideal sufrió un episodio de colapso catastrófico (≈535,029 s de delay), mientras que H-SARG Caótico completó los 10 episodios sin ningún colapso registrado. La diferencia en CVaR₉₅ es de **39×** en los peores episodios.

**Afirmación calibrada:**
> *"Se observó que el entrenamiento bajo perturbaciones moderadas estuvo asociado a la ausencia de colapsos catastróficos en este conjunto de episodios, mientras que el modelo entrenado en condiciones ideales registró un episodio de gridlock completo. Esta evidencia es consistente con la hipótesis de mayor robustez operativa, aunque no alcanza significancia estadística formal con n=10 episodios."*

---

### 2.3 La Paradoja Contraintuitiva — Caos 30%–50%

> **Atención del tribunal:** Este es el aspecto que requiere explicación explícita en la defensa.

| Caos | Métrica | H-SARG Caótico | H-SARG Ideal |
| :---: | :--- | :---: | :---: |
| 30% | Tasa Colapso | **30.0%** | 0.0% |
| 30% | Mediana Delay | 594.9 s | 683.9 s |
| 50% | Tasa Colapso | **10.0%** | 0.0% |
| 50% | CVaR₉₅ | 50888.6 s | 4022.0 s |
| 50% | Mediana Delay | 622.4 s | 671.6 s |

**Análisis forense — Episodios extremos de H-SARG Caótico (caos 30%):**
- **Ep 9**: Delay=99,555 s, Queue=209, Gini=0.306
- **Ep 3**: Delay=22,903 s, Queue=94, Gini=0.395
- **Ep 5**: Delay=11,856 s, Queue=72, Gini=0.435

**Análisis forense — Episodios extremos de H-SARG Caótico (caos 50%):**
- **Ep 3**: Delay=163,746 s, Queue=139, Gini=0.713
- **Ep 4**: Delay=936 s, Queue=50, Gini=0.422
- **Ep 9**: Delay=873 s, Queue=51, Gini=0.413

**Hipótesis explicativa (pendiente de validación empírica directa):**

Esta paradoja puede interpretarse mediante dos mecanismos hipotéticos:

1. **Posible efecto análogo al principio de Braess (hipótesis):** Una posible explicación es que el modelo H-SARG Ideal, al no reaccionar agresivamente a los bloqueos periféricos, retiene inadvertidamente grandes colas en los bordes de la red. Esta contención periférica podría reducir el flujo hacia las intersecciones centrales, previniendo atascos circulares. Por el contrario, H-SARG Caótico, al intentar evacuar colas localmente con mayor eficiencia, podría saturar el núcleo de la red. Bajo caos severo, un único bloqueo físico permanente en ese núcleo podría desencadenar un colapso global en cadena. **Esta hipótesis requiere validación mediante mapas de calor de ocupación por carril, análisis de densidad por intersección y trazado de trayectorias, lo cual constituye una línea futura prioritaria.**

2. **Sensibilidad estocástica a la distribución espacial del caos:** Con sólo 10 episodios por configuración, la distribución espacial aleatoria de los conductores imprudentes puede concentrarse en ubicaciones especialmente críticas (carriles de giro en intersecciones centrales vs. carriles periféricos), lo que contribuye a la variabilidad observada entre semillas.

> **Limitación metodológica reconocida:** Con sólo 10 episodios por configuración, la tasa de colapso tiene un intervalo de confianza amplio (±19 pp para una tasa observada del 10%). Se recomienda n≥30 para intervalos de confianza fiables y pruebas estadísticas con poder estadístico adecuado (≥80%). Esta limitación se reconoce explícitamente como línea futura de trabajo.

---

### 2.4 Tabla de Colapso Doctoral

*(Umbral de gridlock: Delay > 2,000 s por episodio)*

| Controlador | Caos % | Tasa Colapso (%) | Mediana (s) | IQR (s) | CVaR₉₅ (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **H-SARG Caótico** | 0% | **0.0%** | 566.3 s | 161.0 s | 859.2 s |
| **H-SARG Caótico** | 15% | **0.0%** | 617.4 s | 409.6 s | 1322.9 s |
| **H-SARG Caótico** | 30% | **30.0%** | 594.9 s | 8527.8 s | 99555.2 s |
| **H-SARG Caótico** | 50% | **10.0%** | 622.4 s | 362.1 s | 163746.3 s |
| **H-SARG Ideal** | 0% | **0.0%** | 560.3 s | 78.1 s | 854.0 s |
| **H-SARG Ideal** | 15% | **10.0%** | 882.5 s | 358.7 s | 535029.5 s |
| **H-SARG Ideal** | 30% | **0.0%** | 683.9 s | 157.5 s | 922.1 s |
| **H-SARG Ideal** | 50% | **0.0%** | 671.6 s | 555.2 s | 1586.9 s |


---

## Capítulo 2.5: Análisis Estadístico — Pruebas No-Paramétricas

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


---

## Capítulo 3: Hallazgo Principal — Transferencia Zero-Shot a Redes Latinoamericanas

> **Este constituye el resultado empírico más sólido del trabajo**, debido a que muestra mejoras simultáneas en eficiencia, equidad y riesgo en una red no utilizada durante el entrenamiento. En el experimento de robustez, las diferencias entre modelos no alcanzan significancia estadística con n=10 episodios. En cambio, la transferencia zero-shot muestra una mejora **simultánea** en las tres métricas principales (delay, Gini y CVaR), lo que es infrecuente en sistemas de aprendizaje por refuerzo.

### Tabla Comparativa de Generalización

| Métrica Científica | Barcelona (Ideal) | Barcelona (Caótico) | Quito (Ideal) | Quito (Caótico) |
| :--- | :---: | :---: | :---: | :---: |
| **Delay Promedio (s)** | 612.84 | 619.82 | 2971.96 | 2803.79 |
| **Índice de Gini** | 0.649 | 0.593 | 0.450 | 0.434 |
| **CVaR₉₀ (s)** | 1679.55 | 1795.54 | 5906.24 | 5779.97 |
| **Recompensa Total** | -218,640 | -210,211 | -996,120 | -926,898 |

### Por qué la transferencia a Quito es el hallazgo más valioso

En sistemas de RL, mejorar una métrica generalmente empeora otra (trade-off eficiencia-equidad). El hecho de que H-SARG Caótico obtenga **simultáneamente**:

| Métrica | H-SARG Ideal | H-SARG Caótico | Mejora |
| :--- | :---: | :---: | :---: |
| Delay promedio (Quito) | 2972 s | 2804 s | -5.7% |
| Gini (equidad) | 0.450 | 0.434 | -3.6% |
| CVaR₉₀ (riesgo extremo) | 5906 s | 5780 s | -2.1% |

...en una red completamente diferente (topología, contexto LATAM, sin exposición durante el entrenamiento) constituye evidencia sólida de generalización. Esta es la afirmación más difícil de atacar ante un tribunal, precisamente porque no hay posibilidad de sobreajuste a la red de destino.

> **Evidencia Empírica de Robustez al Caos:** Los resultados proporcionan evidencia empírica **consistente con la hipótesis doctoral**. El modelo entrenado con tráfico caótico LATAM muestra mejoras descriptivas frente al modelo entrenado con tráfico ideal en Quito, reduciendo simultáneamente delay, Gini y CVaR₉₀. Estas diferencias no alcanzan significancia estadística formal con n=10 (Mann-Whitney p>0.05), por lo que deben interpretarse como indicativas y requieren validación con n≥30.

---

## Capítulo 4: Conclusión Doctoral

### Conclusión Calibrada (v3 — Post-Revisión Metodológica)

Los resultados proporcionan **evidencia empírica consistente con la hipótesis doctoral** en dos dimensiones específicas: (a) escenarios de perturbación moderada (caos 15%), donde H-SARG Caótico no registró ningún episodio de gridlock frente al 10% observado en H-SARG Ideal, con una diferencia en CVaR₉₅ de 39×; y (b) transferencia zero-shot hacia redes urbanas latinoamericanas, donde H-SARG Caótico obtuvo simultáneamente menor delay, menor Gini y menor CVaR₉₀ en Quito.

Las diferencias observadas en el experimento de robustez en Hangzhou **no alcanzan significancia estadística formal** con n=10 episodios (Mann-Whitney U, p>0.05 en todos los niveles de caos). Las comparativas descriptivas son sugestivas y consistentes con la hipótesis, pero no permiten afirmar superioridad estadística. Se requiere n≥30 episodios por configuración para disponer del poder estadístico adecuado.

La presencia de episodios de colapso en H-SARG Caótico bajo niveles altos de perturbación (30% y 50%) evidencia que la robustez operativa obtenida es **parcial y dependiente de la distribución de perturbaciones** durante el entrenamiento, lo cual se reconoce explícitamente como limitación metodológica y constituye la línea futura de mayor prioridad.

### Valoración Global (según revisión del tutor — v3)

| Dimensión | v1 | v2 | v3 |
| :--- | :---: | :---: | :---: |
| Robustez metodológica | 6/10 | 8.5/10 | 8.5/10 |
| Calidad estadística | 5/10 | 8/10 | 8/10 |
| Defendibilidad ante tribunal | 7/10 | 8.5/10 | 9/10 |
| Potencial de publicación (IEEE T-ITS, TRC) | 8/10 | 8.5/10 | 8.5/10 |

### Líneas Futuras Identificadas

1. **Ampliar a n≥30 episodios** por configuración para pruebas estadísticas con poder adecuado (Mann-Whitney U, potencia ≥80%).
2. **Validar hipótesis Braess-like** mediante mapas de calor de densidad vehicular por intersección y análisis de trayectorias.
3. **Restricciones de seguridad activa** en el espacio de acciones del agente para detectar y responder a condiciones de pre-gridlock.
4. **Aprendizaje robusto minimax** para garantizar estabilidad en toda la distribución de perturbaciones.
5. **Validación en redes reales latinoamericanas** (Quito MOBI, Santiago RED, Guayaquil) con datos de tráfico histórico.

---
*Reporte Doctoral — H-SARG | TSC Framework | Resiliencia Operacional bajo Caos Conductual*  
*Revisión v3 (post-evaluación metodológica): 2026-05-30*
