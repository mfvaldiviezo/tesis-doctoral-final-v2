# src/data_pipeline/__init__.py
"""
Módulo de Ingesta y Preprocesamiento de Datos
=============================================
Responsable de:
- Cargar datasets de tráfico (CSV, XML, sensores loops)
- Limpieza, normalización e imputación
- Ajuste de distribuciones marginales para el módulo de cópulas
- Exportar datos procesados a data/processed/

Clases a implementar (Fase 1+):
    DataLoader      - Cargador de datos multi-fuente
    DataCleaner     - Pipeline de limpieza y validación
    MarginalFitter  - Ajuste de distribuciones univariadas
"""
