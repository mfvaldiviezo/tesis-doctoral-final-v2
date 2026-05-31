# run_all_eval.ps1 - Suite de Evaluacion Doctoral
$ErrorActionPreference = "Continue"

$env:GIT_CONFIG_NOSYSTEM=1
$env:GIT_CONFIG_GLOBAL='NUL'
$env:GIT_CONFIG_SYSTEM='NUL'
$env:PYTHONIOENCODING='utf-8'

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " INICIANDO SUITE DOCTORAL DE EVALUACION UNIFICADA" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# 1. Experimento A (Robustez en Hangzhou)
Write-Host " [1/4] Ejecutando Experimento A: Robustez en Hangzhou (Niveles: 0, 15, 30, 50)..." -ForegroundColor Yellow
baselines\pytsc\venv_pytsc\Scripts\python.exe tsc_framework/scripts/run_evaluation.py --network hangzhou --models ppo_ideal ppo_chaos --baselines fixed maxpressure colight --chaos-levels 0.0 0.15 0.30 0.50 --episodes 10 --output metrics_summary.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host " Error en Experimento A. Abortando suite." -ForegroundColor Red
    exit 1
}

# 2. Experimento B (Transferencia Zero-Shot)
Write-Host " [2/4] Ejecutando Experimento B: Transferencia Zero-Shot (Barcelona y Quito)..." -ForegroundColor Yellow
baselines\pytsc\venv_pytsc\Scripts\python.exe tsc_framework/scripts/transfer_eval.py

if ($LASTEXITCODE -ne 0) {
    Write-Host " Error en Experimento B. Abortando suite." -ForegroundColor Red
    exit 1
}

# 3. Generacion de Figuras (5 figuras con medianas e IQR)
Write-Host " [3/4] Generando Figuras Cientificas (300 DPI, Medianas + IQR)..." -ForegroundColor Yellow
baselines\pytsc\venv_pytsc\Scripts\python.exe tsc_framework/scripts/generate_plots.py

if ($LASTEXITCODE -ne 0) {
    Write-Host " Error al generar graficos." -ForegroundColor Red
}

# 3.5 Analisis Estadistico (Mann-Whitney U, Kruskal-Wallis, Effect Sizes)
Write-Host " [3.5/4] Ejecutando Analisis Estadistico (Mann-Whitney U, Kruskal-Wallis)..." -ForegroundColor Yellow
baselines\pytsc\venv_pytsc\Scripts\python.exe tsc_framework/scripts/statistical_analysis.py

if ($LASTEXITCODE -ne 0) {
    Write-Host " Error en analisis estadistico." -ForegroundColor Red
}

# 4. Consolidacion de Reporte Final
Write-Host " [4/4] Consolidando Reporte Doctoral Final (INFORME_FINAL_ROBUSTEZ.md)..." -ForegroundColor Yellow
baselines\pytsc\venv_pytsc\Scripts\python.exe tsc_framework/scripts/generate_thesis_report.py

if ($LASTEXITCODE -ne 0) {
    Write-Host " Error al consolidar reporte." -ForegroundColor Red
}

Write-Host "=========================================================" -ForegroundColor Green
Write-Host " SUITE COMPLETADA CON EXITO!" -ForegroundColor Green
Write-Host " Ubicacion del reporte: tsc_framework/outputs/results/INFORME_FINAL_ROBUSTEZ.md" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
