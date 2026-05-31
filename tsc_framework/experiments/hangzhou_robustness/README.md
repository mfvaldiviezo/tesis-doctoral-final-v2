# Experimento de Robustez: RL vs Conductores Imprudentes

## Descripción

Este experimento evalúa cómo responden los agentes de Aprendizaje por Refuerzo (RL) entrenados en condiciones ideales cuando se exponen a conductores con comportamiento imprudente basado en datos reales de Quito, Ecuador.

## Hipótesis

**Los agentes RL entrenados en condiciones ideales muestran degradación significativa de rendimiento y equidad cuando se exponen a conductores con comportamiento imprudente típico de Latinoamérica.**

## Objetivos

1. **Cuantificar la caída de rendimiento**: Medir cómo aumentan el retraso promedio y disminuye el throughput a medida que aumenta el porcentaje de conductores imprudentes (0%, 15%, 30%, 50%).

2. **Evaluar impacto en equidad**: Analizar cómo cambian el Índice de Gini y el CVaR (Conditional Value at Risk) con diferentes niveles de imprudencia.

3. **Comparar estrategias**: Contrastar el rendimiento de un agente RL contra un baseline de tiempo fijo.

4. **Validar contexto latinoamericano**: Demostrar que los modelos tradicionales fallan en capturar las características del tráfico en ciudades latinoamericanas.

## Archivos del Experimento

```
experiments/hangzhou_robustness/
├── config_experiment.json       # Configuración del experimento
├── run_experiment.py            # Script principal de ejecución
├── README.md                    # Esta documentación
└── scenarios/                   # Escenarios generados (auto-generado)
    ├── hangzhou_imprudent_00pct.rou.xml
    ├── hangzhou_imprudent_15pct.rou.xml
    ├── hangzhou_imprudent_30pct.rou.xml
    └── hangzhou_imprudent_50pct.rou.xml
```

## Requisitos Previos

1. **Datos de conductores imprudentes generados**:
   ```bash
   python scripts/generate_imprudent_drivers_simple.py \
     --data-path data/quito_behavior/micro_behavior.csv \
     --output-dir results/quito_scenarios \
     --n-scenarios 1000
   ```

2. **Configuración de SUMO para Hangzhou**:
   - `sumo_configs/hangzhou/hangzhou.net.xml`
   - `sumo_configs/hangzhou/hangzhou.rou.xml`
   - `sumo_configs/hangzhou/hangzhou.sumocfg`

3. **Dependencias Python**:
   ```bash
   pip install matplotlib numpy pandas
   ```

## Ejecución del Experimento

### Opción 1: Ejecutar experimento completo

```bash
cd experiments/hangzhou_robustness

python run_experiment.py \
  --config config_experiment.json \
  --output-dir results/hangzhou_robustness_experiment
```

### Opción 2: Generar solo escenarios

```bash
python ../../scripts/inject_imprudent_traffic.py \
  --net-file ../../sumo_configs/hangzhou/hangzhou.net.xml \
  --route-file ../../sumo_configs/hangzhou/hangzhou.rou.xml \
  --imprudent-file ../../results/quito_scenarios/imprudent_drivers.rou.xml \
  --output-dir scenarios \
  --mix-ratio 0.3 \
  --scenario-name imprudent_30pct
```

### Opción 3: Ejecutar con configuración personalizada

Crear un nuevo archivo de configuración:
```bash
cp config_experiment.json config_custom.json
```

Editar los parámetros deseados (ej. más episodios, diferentes ratios) y ejecutar:
```bash
python run_experiment.py \
  --config config_custom.json \
  --output-dir results/custom_experiment
```

## Configuración

El archivo `config_experiment.json` permite personalizar:

### Parámetros de Red
- `net_file`: Ruta al archivo .net.xml
- `route_file`: Ruta al archivo .rou.xml original

### Tráfico Imprudente
- `source_file`: Archivo con vehículos imprudentes generados
- `mix_ratios`: Lista de proporciones a evaluar [0.0, 0.15, 0.30, 0.50]
- `seed`: Semilla para reproducibilidad

### Simulación
- `steps_per_episode`: Duración de cada episodio (default: 3600 steps = 10 horas con delta_time=10)
- `delta_time`: Frecuencia de decisión del agente (default: 10 segundos)
- `n_episodes_per_scenario`: Repeticiones por escenario (default: 5)
- `agent_type`: Tipo de agente ('random', 'rl_trained')

### Métricas
- **Primarias**: average_delay, throughput, gini_index, cvar_risk
- **Secundarias**: queue_length_max, waiting_time_total, co2_emissions

## Resultados Esperados

Al finalizar el experimento, se generarán los siguientes archivos en el directorio de salida:

### Datos Numéricos
- `metrics_summary.csv`: Todas las métricas por episodio y escenario
- `experiment_config_used.json`: Copia de la configuración utilizada

### Visualizaciones
- `plots/performance_drop.png`: Gráfico de retraso y throughput vs % imprudencia
- `plots/fairness_metrics.png`: Barras comparativas de Gini y CVaR
- `plots/scenario_summary.json`: Resumen estadístico por escenario

### Escenarios Generados
- `scenarios/hangzhou_imprudent_XXpct.rou.xml`: Archivos de rutas con mezcla de tráfico
- `scenarios/imprudent_XXpct_summary.json`: Metadata de cada escenario

## Interpretación de Resultados

### Métricas Clave

1. **Average Delay (s)**: Tiempo promedio de espera por vehículo. 
   - *Esperado*: Aumenta con % de imprudencia

2. **Throughput (vehículos)**: Número de vehículos que completan su viaje.
   - *Esperado*: Disminuye con % de imprudencia

3. **Gini Index (0-1)**: Medida de desigualdad en tiempos de espera.
   - *Esperado*: Aumenta (mayor inequidad) con imprudencia

4. **CVaR Risk**: Riesgo de cola (promedio de peores casos).
   - *Esperado*: Aumenta significativamente con imprudencia

### Validación de Hipótesis

La hipótesis se considera validada si:
- ✅ El delay aumenta >20% al pasar de 0% a 50% de imprudencia
- ✅ El throughput disminuye >15% en el mismo rango
- ✅ El índice Gini muestra aumento consistente
- ✅ El CVaR incrementa más que el delay promedio (indicando eventos extremos)

## Siguientes Pasos

### Fase 2: Entrenamiento de Agente Robusto

Una vez validada la hipótesis con agentes baseline:

1. **Entrenar nuevo agente** expuesto a conductores imprudentes:
   ```bash
   python train_robust_agent.py \
     --scenario-dir results/hangzhou_robustness_experiment/scenarios \
     --mix-ratio-range 0.3-0.5 \
     --episodes 10000
   ```

2. **Re-evaluar** el agente robusto en el mismo experimento:
   ```bash
   python run_experiment.py \
     --config config_experiment.json \
     --agent-checkpoint models/robust_agent.pth
   ```

3. **Comparar** rendimiento entre agente baseline y robusto.

### Fase 3: Transferencia a Ciudades Latinoamericanas

1. Obtener topologías de ciudades latinas (Quito, Bogotá, Lima, etc.)
2. Adaptar configuraciones de SUMO
3. Repetir experimentos para validar transferibilidad

## Solución de Problemas

### Error: "No se encontró el archivo .net.xml"
Verifica que los archivos de Hangzhou estén en `sumo_configs/hangzhou/`. Si no existen, descárgalos del repositorio oficial de SUMO o genera una red de prueba.

### Error: "Faltan columnas en data"
Asegúrate de haber generado primero los escenarios imprudentes con `generate_imprudent_drivers_simple.py`.

### Error: "SUMO no encontrado"
Verifica que SUMO esté instalado y en el PATH:
```bash
sumo --version
```

## Referencias

- Dataset PoliDriving: Telemetría Microscópica Ecuador
- Framework TSC: Traffic Signal Control con fairness
- Cópulas Vine: Modelado de dependencias multivariadas
- SUMO Simulator: Simulation of Urban MObility

## Autor

Experimento desarrollado como parte de investigación de tesis sobre optimización de tráfico con IA en contextos latinoamericanos.
