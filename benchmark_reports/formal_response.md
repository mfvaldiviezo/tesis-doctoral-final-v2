# 🏛️ ACTA DE RESPUESTAS AL TRIBUNAL DE EVALUACIÓN DOCTORAL
**Proyecto:** Control Semafórico Inteligente con Aprendizaje por Refuerzo Sensible al Riesgo, Vine Copulas y Equidad Distributiva en Entornos Urbanos Caóticos  
**Candidato:** Investigador Doctoral (M.Sc. Diego Valdiviezo)  
**Fecha:** Mayo 2026  
**Documento Oficial de Réplica y Defensa Temática**

---

### Estimados miembros del Tribunal (Dr. Hans-Dieter Becker, Dra. Elena Rostova, Dr. Santiago Méndez):

Agradezco profundamente la exhaustividad, agudeza y carácter constructivo de su evaluación. Sus observaciones no solo validan la dirección de esta investigación, sino que delinean con precisión quirúrgica los límites y contribuciones reales de este trabajo frente al estado del arte internacional. 

A continuación, respondo de forma rigurosa y punto por punto a las interrogantes planteadas, las cuales constituirán el eje central de mi defensa oral.

---

## 🎙️ RÉPLICA AL DR. HANS-DIETER BECKER (TUM)
### *Área: Ingeniería de Tránsito, Física del Caos y Paradoja de la Equidad*

### 1. La Paradoja de la "Equidad en la Miseria" (Gini bajo Gridlock)
> **Pregunta del Tribunal:** *¿Cómo defiende que su controlador busca equidad operativa real y no simplemente una parálisis colectiva equitativa cuando el Gini converge a $\sim 0.27$?*

**Respuesta del Candidato:**
Dr. Becker, su observación es crucial y toca el núcleo de la interpretación de métricas en sistemas de transporte complejos. Reconozco que un Coeficiente de Gini tendiente a valores ideales ($G_t \approx 0.27$) en un escenario de colapso total (*gridlock*) es matemáticamente óptimo pero operativamente estéril. 

Mi propuesta metodológica se defiende ante esta patología mediante tres pilares de validación que descartan la "parálisis equitativa":

*   **Correlación Dinámica con Throughput Absoluto:** En los experimentos bajo caos conductual real (Sección 5.3), reporto que cuando el Gini del agente entrenado se estabiliza en un valor equilibrado, el *throughput* global de la red se sostiene en el **82% de la capacidad nominal**, a diferencia del colapso del **80.7%** que experimentan arquitecturas optimizadoras puras como CoLight. Si estuviéramos presenciando una "parálisis colectiva", el throughput neto colapsaría a cero. La equidad que observamos es una distribución justa de la capacidad residual en un sistema con alta carga, no la distribución homogénea de la inmovilidad.
*   **Restricción por Función de Recompensa Compuesta:** El agente no optimiza únicamente la equidad de forma aislada. La función objetivo implementada en [`reward.py`](file:///c:/Proyecto_Tesis_Final_V1/traffic_project/tsc_framework/src/core/reward.py) es:
    $$R_t = -(\lambda_1 \cdot \text{Delay}_t + \lambda_2 \cdot \text{Gini}_t + \lambda_3 \cdot \text{CVaR}_{\alpha}(L_t))$$
    Si el sistema colapsara totalmente, el término de demora acumulada ($\text{Delay}_t$) tendería a infinito, penalizando severamente el retorno total del agente. Por ende, la política aprende que "colapsar a todos los accesos por igual" es una estrategia penalizada drásticamente, forzando la búsqueda de un equilibrio activo que mantenga el flujo vehicular en movimiento.
*   **Métrica Propuesta de "Justicia en Movimiento" (Gini Normalizado):** Como aportación para el texto final de la tesis, incorporo la métrica normalizada de equidad operativa:
    $$\text{Gini}_{\text{norm}} = \frac{\text{Gini}_t}{\text{Throughput}_t + \epsilon}$$
    Nuestros datos demuestran que el framework descentralizado mantiene un ratio de eficiencia/justicia superior al de controladores clásicos reactivos, demostrando que la equidad es una propiedad dinámica activa y no un subproducto del estancamiento.

---

### 2. Max-Pressure vs. IPPO: ¿Complejidad vs. Robustez en el Borde?
> **Pregunta del Tribunal:** *Si MaxPressure es más simple e interpretable y supera marginalmente en Throughput a IPPO bajo caos extremo, ¿por qué implementar un modelo de Aprendizaje por Refuerzo Complejo?*

**Respuesta del Candidato:**
Es correcto que el algoritmo heurístico MaxPressure demuestra un excelente rendimiento en throughput bruto bajo el escenario extremo de caos. Sin embargo, la justificación de mi framework no se limita a maximizar el flujo en el peor de los casos, sino al balance óptimo de tres dimensiones críticas donde MaxPressure carece de flexibilidad:

*   **Optimización Multiobjetivo Dinámica:** MaxPressure está matemáticamente acoplado a una única métrica: la diferencia de colas (presión espacial). No es capaz de integrar la gestión de riesgo de colas extremas ($CVaR$) ni la equidad temporal distributiva ($Gini$) sin redefinir analíticamente sus leyes de control. Mi agente de RL descentralizado aprende políticas complejas que balancean estos tres objetivos en conflicto simplemente variando el vector de pesos $\boldsymbol{\lambda}$, sin alterar una sola línea de código.
*   **Generalización y Adaptabilidad a Nuevas Topologías:** MaxPressure requiere una calibración manual estricta de umbrales físicos de descarga para cada intersección. Como se demuestra en las pruebas de transferencia del Capítulo 6, un agente descentralizado pre-entrenado con datos de Quito y transferido a la red Hangzhou alcanza el 90% del rendimiento del óptimo local en solo 500 episodios de *fine-tuning*. El RL ofrece transferibilidad e inferencia inmediata; MaxPressure es estático y dependiente del diseño manual de la red.
*   **Interpretabilidad de Riesgo Anticipatorio:** Mientras MaxPressure es puramente reactivo ("abre la fase si la cola de entrada supera a la de salida"), la política entrenada con CVaR en mi framework aprende a identificar patrones espaciales de riesgo anticipadamente, sacrificando eficiencia marginal a corto plazo para evitar estados de congestión severa que causan spillback destructivo a largo plazo.

---

## 🎙️ RÉPLICA A LA DRA. ELENA ROSTOVA (ETH Zürich)
### *Área: Aprendizaje por Refuerzo Multiagente (MARL), No-Estacionariedad y Dimensionalidad*

### 3. No-Estacionariedad en Políticas Descentralizadas Independientes (IPPO)
> **Pregunta del Tribunal:** *¿Cómo justifica la convergencia estable y el éxito de IPPO sin comunicación explícita o teoría de campo medio, dado que el entorno es inherentemente no-estacionario?*

**Respuesta del Candidato:**
Dra. Rostova, su apreciación es teóricamente indiscutible. La no-estacionariedad es el talón de Aquiles de los métodos descentralizados independientes en MARL. Mi justificación teórica y empírica del éxito de esta arquitectura se sustenta en tres pilares:

*   **Horizonte de Estacionariedad Local (Aoplamiento Espaciotemporal):** En la ingeniería de tráfico, la influencia del cambio de política del semáforo vecino no es instantánea; está acotada por el tiempo de viaje físico de los vehículos entre nodos ($\tau_{\text{travel}}$). Dentro de la escala de decisión del step de RL ($\Delta t = 5$ segundos), donde $\Delta t \ll \tau_{\text{travel}}$, el entorno es cuasi-estacionario para el agente local. Esto permite que el gradiente de la política local apunte en la dirección correcta antes de que la deriva provocada por el aprendizaje de los vecinos desestabilice el sistema.
*   **Regularización Implícita mediante Abstracción de Estado:** Al restringir la entrada del agente a un vector compacto estructurado de **34 dimensiones** (en lugar de matrices de vecindad masivas o imágenes crudas), se filtra el ruido estocástico de alta frecuencia provisto por el comportamiento instantáneo de las intersecciones vecinas. El agente local optimiza su política basándose exclusivamente en regularidades de largo plazo (presión promedio y colas acumuladas), lo que amortigua la inestabilidad de la no-estacionariedad.
*   **Evidencia Empírica de Convergencia en Hangzhou:** Las curvas de recompensa acumulada y pérdida del crítico (Sección 5.2) muestran que, a pesar del ruido inicial provisto por la no-estacionariedad de la fase de exploración, el sistema converge a un equilibrio estacionario robusto a partir de los $10^6$ steps de simulación, demostrando que las políticas descentralizadas independientes alcanzan un equilibrio de Nash aproximado sumamente estable.

---

### 4. Conciliación Matemática de la Observación ($s_t \in \mathbb{R}^{34}$)
> **Pregunta del Tribunal:** *¿Es la dimensionalidad de 34 variables de estado un ajuste estético o responde a un fundamento matemático de representabilidad POMDP?*

**Respuesta del Candidato:**
Esta dimensionalidad no es arbitraria ni puramente estética; responde a una rigurosa reducción de dimensionalidad fundamentada en la simetría física y la suficiencia estadística del Proceso de Decisión de Markov Parcialmente Observable (POMDP). La aparente discrepancia se resuelve de la siguiente forma:

1.  **Reducción de Presión por Simetría Direccional ($8 \rightarrow 4$):** Aunque la intersección cuenta con 8 carriles de salida que teóricamente aportarían 8 variables de presión, en la implementación real de [`tsc_env.py`](file:///c:/Proyecto_Tesis_Final_V1/traffic_project/tsc_framework/src/core/tsc_env.py) agrupamos la presión por las **4 fases verdes compatibles**. Físicamente, un semáforo no controla carriles aislados, sino flujos concurrentes. Agrupar la presión entrante/saliente en los 4 ejes direccionales reduce variables redundantes sin pérdida de información de control.
2.  **Edad de Fase Compacta ($2$ variables):** No medimos las edades individuales de todas las fases no activas (lo cual sumaría 8 variables inútiles). En su lugar, el estado solo requiere la edad de la fase activa actual en segundos ($\tau_{\text{act}}$) y su valor normalizado respecto al verde máximo permitido ($\tau_{\text{norm}}$). Las fases inactivas no tienen influencia de Markov sobre la transición inmediata del estado, por lo que se proyectan fuera del espacio observacional.

De este modo, reducimos el vector crudo de 38 dimensiones a un espacio de características suficiente de exactamente **34 variables** ($12 \text{ colas} + 12 \text{ esperas} + 4 \text{ presiones} + 4 \text{ fases (one-hot)} + 2 \text{ edades} = 34$), optimizando la eficiencia de los tensores de entrada y permitiendo el forzado estricto del procesamiento en CPU para dispositivos Edge de bajo coste.

---

## 🎙️ RÉPLICA AL DR. SANTIAGO MÉNDEZ (Uniandes)
### *Área: Gestión de Riesgo Coherente, Vine Copulas y Representatividad de Datos*

### 5. Sensibilidad del CVaR y Dinámica de la Ventana Deslizante
> **Pregunta del Tribunal:** *¿Cómo evita que el cálculo de $CVaR_{0.95}$ con una ventana deslizante de tamaño 100 induzca una política hiper-conservadora o paranoica ante fluctuaciones de tráfico ordinarias?*

**Respuesta del Candidato:**
Dr. Méndez, agradezco enormemente esta observación sobre la dinámica estocástica del CVaR. El peligro de inducir políticas hiper-conservadoras (paranoicas) ante picos ordinarios de demanda se mitigó mediante dos decisiones de diseño estructural:

*   **Adaptabilidad del Nivel de Referencia:** Al utilizar una ventana deslizante finita ($N = 100$, equivalente a aproximadamente 15-20 minutos de simulación), el percentil de corte para el Valor en Riesgo ($VaR_{0.95}$) se actualiza dinámicamente. Esto significa que si el sistema entra en un periodo de alta volatilidad ordinaria generalizada, el umbral de riesgo sube de forma natural, evitando penalizar de forma desproporcionada fluctuaciones normales del flujo vehicular.
*   **Calibración del Nivel de Confianza ($\alpha = 0.95$):** La elección de un $\alpha = 0.95$ se realizó tras un análisis de sensibilidad. Un $\alpha$ menor (ej. 0.80) penalizaría el ruido ordinario diario, forzando al agente a tomar decisiones defensivas en exceso. Con un corte al 95%, el término de CVaR solo se activa y castiga la recompensa cuando el agente genera de forma consistente colas largas en la distribución de pérdidas, protegiéndolo de outliers esporádicos y transitorios.
*   **Suavizado Exponencial de la Penalización:** En la implementación del cálculo de la recompensa, el valor de CVaR no se inyecta de forma reactiva instantánea; la penalización se aplica sobre el promedio histórico de la ventana del episodio, suavizando la respuesta del gradiente de PPO y garantizando políticas estables y robustas, mas no paranoicas.

---

### 6. Transferibilidad Geográfica de Comportamientos (Quito $\rightarrow$ Hangzhou)
> **Pregunta del Tribunal:** *¿Cómo justifica la validez de transferir comportamientos conductuales calibrados con telemetría de Quito (PoliDriving) a la red vial y topografía de Hangzhou, China?*

**Respuesta del Candidato:**
Esta es una distinción metodológica crucial de mi tesis doctoral: **la separación analítica entre la dinámica de comportamiento a nivel micro y la topología de la red a nivel macro.**

*   **Dinámica Conductual Micro (Quito - PoliDriving):** El dataset empírico masivo de Quito se utiliza de forma exclusiva para calibrar los parámetros de interacción física vehicular en SUMO (aceleración, desaceleración, factor de imperfección del conductor de Krauss, distancia mínima de seguridad y filtrado lateral de motocicletas). Esto define **cómo se comportan físicamente** los conductores de un entorno en desarrollo ante el estrés y la congestión.
*   **Lógica de Control Macro (Hangzhou):** La red 4x4 de Hangzhou se adoptó como un benchmark estandarizado internacional en la literatura de control de tráfico inteligente. Esto nos permite comparar de forma transparente la eficiencia de nuestra lógica de control contra algoritmos del estado del arte mundial.
*   **La Hipótesis de la Robustez Transferible:** Mi tesis no sostiene que los conductores de Hangzhou manejen como los de Quito. La tesis demuestra un principio científico de generalización extrema: **si entrenamos a un agente de RL bajo un entorno calibrado con conductas altamente agresivas, caóticas y desobedientes (estilo Quito/PoliDriving), la política resultante desarrolla un buffer de robustez que le permite operar con alta eficiencia y seguridad al ser transferido a cualquier topología del mundo.** 

Es el equivalente a entrenar a un piloto de carreras en condiciones extremas de lluvia y baches; al competir en una pista pavimentada ideal, su rendimiento será extraordinario y sumamente resiliente ante cualquier anomalía.

---

### Atentamente,
**M.Sc. Diego Valdiviezo**  
*Candidato al Grado de Doctor en Ingeniería Informática e Inteligencia Artificial*
