# src/copulas/__init__.py
"""
Módulo Probabilístico - Vine Copulas
=====================================
Responsable de:
- Modelar dependencias multivariadas entre flujos de tráfico
- Generar escenarios de estrés mediante simulación de cópulas
- Exportar escenarios como rutas SUMO (.rou.xml) o arrays NumPy

Clases a implementar (Fase 2):
    VineCopulaModel       - Ajuste del modelo C/D/R-Vine con pyvinecopulib
    StressScenarioGen     - Generador de N escenarios en el cuantil alpha
    RouteFileExporter     - Conversión de escenarios a archivos SUMO .rou.xml
"""
