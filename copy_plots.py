import shutil
import os

source_dir = r"c:\Proyecto_Tesis_Final_V1\traffic_project\benchmark_reports\plots"
dest_dir = r"C:\Users\mfval\.gemini\antigravity-ide\brain\58201c31-2a01-4cd5-bbe6-2bb55e391fd1"

files = ["efficiency_vs_equity.png", "resilience_throughput.png"]

print("Copiando archivos...")
for f in files:
    src = os.path.join(source_dir, f)
    dst = os.path.join(dest_dir, f)
    try:
        shutil.copy(src, dst)
        print(f"Copiado: {f} -> {dst}")
    except Exception as e:
        print(f"Error al copiar {f}: {e}")
