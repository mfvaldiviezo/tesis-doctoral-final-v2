# TSC Framework — Control Semafórico Inteligente

> **Tesis Doctoral**: Framework modular para control semafórico inteligente basado en
> Aprendizaje por Refuerzo sensible al riesgo, Vine Copulas y métricas de equidad distributiva.

---

## 📁 Estructura del Proyecto

```
tsc_framework/
├── environment.yml          # Entorno Conda reproducible
├── setup.py                 # Instalación local vía pip install -e .
├── config/
│   └── default_config.yaml  # Hiperparámetros, rutas SUMO, semillas
├── data/
│   ├── raw/                 # Datos originales (NO modificar)
│   └── processed/           # Datos limpios + marginales ajustadas
├── sumo_configs/
│   ├── networks/            # Archivos .net.xml (redes de tráfico)
│   └── routes/              # Archivos .rou.xml (flujos vehiculares)
├── src/
│   ├── data_pipeline/       # Ingesta y preprocesamiento
│   ├── copulas/             # Generador de escenarios (Vine Copulas)
│   ├── rl_env/              # Entorno Gymnasium + TraCI
│   ├── agents/              # PPO/SAC con CVaR y Gini
│   └── utils/               # Métricas, logging, config loader
├── scripts/
│   ├── train.py             # Entrenamiento del agente
│   └── evaluate.py          # Evaluación y pruebas de estrés
├── tests/                   # Pruebas unitarias (pytest)
└── outputs/                 # Modelos, logs, resultados (generado en runtime)
```

---

## 🚀 Inicio Rápido

### 1. Crear el entorno Conda

```bash
conda env create -f environment.yml
conda activate tsc-env
```

### 2. Instalar el paquete local en modo editable

```bash
pip install -e .
```

### 3. Verificar el andamiaje

```bash
pytest tests/ -v
```

### 4. Configurar SUMO_HOME

En Windows (PowerShell):
```powershell
$env:SUMO_HOME = "C:\Program Files\Eclipse\Sumo"
```

O edita `config/default_config.yaml`:
```yaml
sumo:
  sumo_home: "C:/Program Files/Eclipse/Sumo"
```

---

## 🔬 Componentes del Framework

| Módulo | Fase | Descripción |
|--------|------|-------------|
| `data_pipeline` | 1 | Ingesta de datos de tráfico, ajuste de marginales |
| `rl_env` | 1 | Entorno Gymnasium con control TraCI sobre SUMO |
| `copulas` | 2 | Vine Copulas para generación de escenarios de estrés |
| `agents` | 3 | PPO/SAC con penalización CVaR y coeficiente Gini |
| `utils` | 0 | Utilidades transversales: métricas, logging, config |

---

## ⚙️ Configuración

Todos los hiperparámetros se gestionan desde `config/default_config.yaml`.
Los valores más relevantes:

```yaml
reproducibility:
  global_seed: 42

agent:
  algorithm: "PPO"
  total_timesteps: 1_000_000

risk_metrics:
  enable_cvar: true
  cvar_alpha: 0.95
  enable_gini: true
```

---

## 📦 Dependencias Clave

- **Eclipse SUMO** ≥ 1.19 — Simulador de tráfico
- **Gymnasium** ≥ 0.29 — API de entornos RL
- **Stable-Baselines3** ≥ 2.2.1 — Implementaciones PPO/SAC/TD3
- **pyvinecopulib** ≥ 0.6.3 — Modelado con Vine Copulas
- **SHAP** ≥ 0.44 — Explicabilidad del agente

---

## 📄 Licencia

Proyecto académico — Tesis Doctoral. Todos los derechos reservados.
