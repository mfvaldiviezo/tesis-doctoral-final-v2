$eff = Get-Content "c:\Proyecto_Tesis_Final_V1\traffic_project\efficiency_base64.txt" -Raw
$res = Get-Content "c:\Proyecto_Tesis_Final_V1\traffic_project\resilience_base64.txt" -Raw
$tmpl = Get-Content "c:\Proyecto_Tesis_Final_V1\traffic_project\template_report.md" -Raw

$tmpl = $tmpl.Replace("{{EFFICIENCY_PLOT}}", $eff).Replace("{{RESILIENCE_PLOT}}", $res)

[System.IO.File]::WriteAllText("c:\Proyecto_Tesis_Final_V1\traffic_project\tsc_framework\REPORTE_RESULTADOS_DOCTORADO.md", $tmpl, [System.Text.Encoding]::UTF8)
Write-Output "Reporte generado con exito."
