# CAPÍTULO 4.3.2: EVALUACIÓN DE RESILIENCIA MULTI-AGENTE BAJO ESTRÉS VIAL LATAM

## 1. INTRODUCCIÓN Y CONTEXTO METODOLÓGICO
Para validar la robustez de los algoritmos de control semafórico inteligente en escenarios de alta incertidumbre y comportamiento subóptimo, se sometió a cuatro arquitecturas representativas (Fixed Time, Max Pressure, IPPO y CoLight) a una prueba de esfuerzo basada en dinámicas caóticas de conducción (LatamChaos). La inyección física del caos incluyó:
*   **Reducción del intervalo de seguridad vial ($T_u = 0.5$ s)** para emular conductas de *tailgating* (alcance y seguimiento agresivo).
*   **Aceleración errática e imprudente ($SpeedFactor \in [1.2, 1.8]$)** y aumento del coeficiente de imperfección del conductor (Krauss $sigma = 0.9$).
*   **Bloqueo intencional de intersecciones (Gridlock)**, deteniendo físicamente los vehículos infractores en el centro geométrico de los cruces para interrumpir los flujos transversales.
*   **Desactivación del modelo preventivo de colisiones** de SUMO, forzando teletransportes y penalizaciones tras accidentes reales de colisión.

A continuación, se discute el comportamiento emergente del sistema bajo las dimensiones clave determinadas en el marco metodológico: Eficiencia (Throughput), Riesgo Operacional (CVaR95) y Equidad Espacial (Coeficiente de Gini).

---

## 2. ANÁLISIS DE DEGRADACIÓN DE LA EFICIENCIA (THROUGHPUT)
El comportamiento macroscópico de los flujos de tráfico ante la introducción de perturbaciones agresivas arrojó variaciones de rendimiento drásticas y contrastantes entre las familias de control evaluadas.

```
       Δ THROUGHPUT (Ideal → Caótico)
       ┌───────────────────────────────┐
FIXED  │ -77.0% 🔴                     │
MP     │ +18.2% 🟡 (Base marginal)     │
IPPO   │ -13.0% 🟡 (Alta resiliencia)  │
CoLight│ -80.7% 🔴 (Colapso total)     │
       └───────────────────────────────┘
```

### 2.1 El Colapso de CoLight: La Vulnerabilidad de la Optimización Codiciosa Global
En condiciones de tráfico ideal, **CoLight** demuestra la máxima eficiencia global alcanzando un throughput de $0.3024$ veh/s. Este rendimiento sobresaliente se debe a su mecanismo de atención de grafos, el cual coordina las fases semafóricas basándose en la demanda de toda la red. 

Sin embargo, ante el estrés caótico de **LatamChaos**, CoLight experimenta un **colapso catastrófico del -80.7%** en su throughput ($0.0583$ veh/s). Al no contemplar la posibilidad de que los vehículos bloqueen intersecciones o sufran accidentes físicos que paralicen las vías, el modelo de aprendizaje por refuerzo espacialmente coordinado toma decisiones basadas en estados saturados que propagan el bloqueo a toda la red. El colapso del flujo es generalizado debido al arrastre de colas hacia atrás (back-spillover effect).

### 2.2 La Resiliencia Emergente de IPPO
Por el contrario, **IPPO** exhibe el comportamiento más robusto en términos relativos, limitando su degradación de throughput a tan solo un **-13.0%** ($0.0957 \rightarrow 0.0832$ veh/s). Aunque su throughput óptimo es menor en condiciones ideales que el de CoLight, el aprendizaje neuronal descentralizado e independiente de IPPO dota a cada semáforo de una política más defensiva y conservadora frente al comportamiento errático local de los vehículos. Al entrenarse de forma aislada, el agente IPPO asume implícitamente un entorno dinámicamente inestable, lo que le permite mantener un flujo vehicular consistente incluso cuando las intersecciones colindantes sufren bloqueos parciales.

---

## 3. LA PARADOJA DE LA EQUIDAD ESPACIAL (GINI TEMPORAL Y FINAL)
Uno de los descubrimientos matemáticos más significativos de este benchmark radica en la evolución del Coeficiente de Gini sobre la distribución temporal y final de colas entre semáforos.

```
       GINI TEMPORAL (Ideal → Caótico)
       ┌───────────────────────────────┐
Ideal  │ FIXED: 0.52 | CoLight: 0.50   │ -> Alta disparidad espacial
Caótico│ FIXED: 0.27 | CoLight: 0.27   │ -> "Equidad en la miseria"
       └───────────────────────────────┘
```

Bajo la física tradicional sin colisiones, el caos vial provocaba un incremento severo de la inequidad (Gini disparándose), dado que las arterias dominantes con conductores agresivos devoraban el tiempo en verde a costa de las transversales. Sin embargo, al habilitar el realismo físico de los accidentes y los bloqueos transversales, **el Coeficiente de Gini experimentó una caída drástica generalizada, estabilizándose en valores muy bajos ($\sim 0.27$)**.

### 3.1 Explicación Física de la "Equidad en la Miseria"
En la teoría de flujo de tráfico en redes urbanas, este fenómeno corresponde al **estado de parálisis sistémica o gridlock generalizado**. Cuando las colisiones y los bloqueos intencionales obstruyen el flujo en las intersecciones clave, la congestión deja de ser un fenómeno localizado. 

La cola de vehículos retrocede hasta ocupar la totalidad de los enlaces de la red Hangzhou 4x4. Como consecuencia:
1.  **Todos** los semáforos, sin importar el algoritmo de control que posean, registran colas máximas y saturación física persistente en todos sus accesos.
2.  La variabilidad espacial y temporal de las colas disminuye drásticamente debido a la inmovilidad del sistema.
3.  El algoritmo matemático de Gini interpreta esta homogeneidad en el estancamiento como una **distribución equitativa del retraso**.

**Lección Doctoral:** La métrica de equidad (Gini) no debe ser analizada de forma aislada en sistemas urbanos. Un índice Gini idealmente bajo ($\sim 0.27$) es patológico si viene acompañado de un colapso del $80\%$ del Throughput, lo que refleja un estado de parálisis colectiva en lugar de una distribución eficiente y justa del recurso verde semafórico.

---

## 4. CONCLUSIONES METODOLÓGICAS PARA EL DISEÑO DE TSC
El benchmark ejecutado aporta tres contribuciones cruciales para el campo del Control Inteligente de Tráfico (TSC) mediante MARL:
1.  **Los modelos de simulación tradicionales "collision-free" sesgan severamente el desarrollo de agentes.** Ignorar los accidentes físicos favorece a las políticas agresivas y optimizadoras globales (como los enfoques de atención de grafos), haciéndolas parecer superiores cuando en el mundo real colapsarían de inmediato.
2.  **IPPO demuestra ser una alternativa de alta confiabilidad vial.** El aprendizaje descentralizado ofrece un buffer de seguridad intrínseco contra perturbaciones estocásticas complejas y comportamientos antisociales de conducción.
3.  **Métricas de Riesgo Operacional (CVaR95):** El comportamiento del riesgo operado sobre colas extremas bajo el caos demuestra que la parálisis colectiva nivela el peor 5% de las colas a lo largo de la simulación, reduciendo el diferencial dinámico y forzando a los sistemas inteligentes a operar en regímenes de contención de daños en lugar de optimización dinámica.
