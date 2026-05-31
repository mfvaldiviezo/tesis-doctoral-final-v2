# run_experiment_local.ps1
# Configurar error para que NO cierre el terminal inmediatamente
$ErrorActionPreference = "Continue"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "INICIANDO EXPERIMENTO LOCAL: BASELINE vs IMPRUDENTES" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Definir Rutas Absolutas
$baseDir = "C:\Proyecto_Tesis_Final_V1\traffic_project\tsc_framework"
$netFile = Join-Path $baseDir "sumo_configs\networks\hangzhou_4x4.net.xml"
$routeDense = Join-Path $baseDir "sumo_configs\routes\hangzhou\hangzhou_dense.rou.xml"
$outputDirScen = Join-Path $baseDir "experiments\hangzhou_robustness\scenarios"
$outputDirRes = Join-Path $baseDir "experiments\hangzhou_robustness\results"
$imprudentFile = Join-Path $baseDir "results\quito_scenarios\imprudent_drivers_fixed.rou.xml"

# Ruta a las herramientas de SUMO (AJUSTA ESTO SI TU INSTALACIÓN ES DIFERENTE)
$sumoTools = "C:\Program Files (x86)\Eclipse\Sumo\tools"
$randomTripsScript = Join-Path $sumoTools "randomTrips.py"

# 2. Crear Directorios
Write-Host "`n[1/6] Creando directorios..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $outputDirScen | Out-Null
New-Item -ItemType Directory -Force -Path $outputDirRes | Out-Null
Write-Host "Directorios listos." -ForegroundColor Green

# 3. Generar Rutas Densas
Write-Host "`n[2/6] Generando tráfico aleatorio denso (hangzhou_dense.rou.xml)..." -ForegroundColor Yellow
if (Test-Path $randomTripsScript) {
    python $randomTripsScript -n $netFile -r $routeDense -e 3600 --fringe-factor 200 --seed 42
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR al generar rutas. Verifica que el archivo .net.xml sea válido." -ForegroundColor Red
        Read-Host "Presiona Enter para salir..."
        exit 1
    }
} else {
    Write-Host "ERROR: No se encontró randomTrips.py en $randomTripsScript" -ForegroundColor Red
    Read-Host "Presiona Enter para salir..."
    exit 1
}

if (Test-Path $routeDense) {
    Write-Host "Rutas densas generadas correctamente." -ForegroundColor Green
} else {
    Write-Host "ERROR: El archivo de rutas no se creó." -ForegroundColor Red
    Read-Host "Presiona Enter para salir..."
    exit 1
}

# 4. Inyectar Tráfico Imprudente (30%)
Write-Host "`n[3/6] Inyectando 30% de conductores imprudentes..." -ForegroundColor Yellow
python (Join-Path $baseDir "scripts\inject_imprudent_traffic.py") `
  --net-file $netFile `
  --route-file $routeDense `
  --imprudent-file $imprudentFile `
  --output-dir $outputDirScen `
  --mix-ratio 0.3 `
  --scenario-name quito_imprudent_30pct_dense

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en inyección de imprudentes." -ForegroundColor Red
    Read-Host "Presiona Enter para salir..."
    exit 1
}

# 5. Inyectar Tráfico Baseline (0%)
Write-Host "`n[4/6] Generando escenario Baseline (0% imprudentes)..." -ForegroundColor Yellow
python (Join-Path $baseDir "scripts\inject_imprudent_traffic.py") `
  --net-file $netFile `
  --route-file $routeDense `
  --imprudent-file $imprudentFile `
  --output-dir $outputDirScen `
  --mix-ratio 0.0 `
  --scenario-name baseline_0pct

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en inyección de baseline." -ForegroundColor Red
    Read-Host "Presiona Enter para salir..."
    exit 1
}

# 6. Ejecutar Simulación BASELINE
Write-Host "`n[5/6] Ejecutando simulación BASELINE (0% imprudentes)..." -ForegroundColor Yellow
$baselineRoute = Join-Path $outputDirScen "hangzhou_baseline_0pct.rou.xml"
$statsBase = Join-Path $outputDirRes "stats_baseline_0pct.xml"
$tripBase = Join-Path $outputDirRes "tripinfo_baseline_0pct.xml"

sumo -n $netFile -r $baselineRoute `
     --no-step-log `
     --statistic-output $statsBase `
     --tripinfo-output $tripBase `
     --end 3600 `
     --quit-on-end

if ($LASTEXITCODE -eq 0) {
    Write-Host "Simulación BASELINE completada con éxito." -ForegroundColor Green
} else {
    Write-Host "ERROR en simulación BASELINE." -ForegroundColor Red
}

# 7. Ejecutar Simulación IMPRUDENTES
Write-Host "`n[6/6] Ejecutando simulación IMPRUDENTES (30%)..." -ForegroundColor Yellow
$imprudentRoute = Join-Path $outputDirScen "hangzhou_quito_imprudent_30pct_dense.rou.xml"
$statsImp = Join-Path $outputDirRes "stats_imprudent_30pct.xml"
$tripImp = Join-Path $outputDirRes "tripinfo_imprudent_30pct.xml"

sumo -n $netFile -r $imprudentRoute `
     --no-step-log `
     --statistic-output $statsImp `
     --tripinfo-output $tripImp `
     --end 3600 `
     --quit-on-end

if ($LASTEXITCODE -eq 0) {
    Write-Host "Simulación IMPRUDENTES completada con éxito." -ForegroundColor Green
} else {
    Write-Host "ERROR en simulación IMPRUDENTES." -ForegroundColor Red
}

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENTO FINALIZADO" -ForegroundColor Cyan
Write-Host "Resultados en: $outputDirRes" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Read-Host "Presiona Enter para cerrar..."