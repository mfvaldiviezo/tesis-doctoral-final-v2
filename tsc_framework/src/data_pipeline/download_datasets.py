import os
import subprocess
import shutil
import logging

# Configuración de logs para la terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas base del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
POLIDRIVING_DIR = os.path.join(RAW_DATA_DIR, "polidriving")
MACRO_DIR = os.path.join(RAW_DATA_DIR, "macro_traffic")
TEMP_REPO_DIR = os.path.join(RAW_DATA_DIR, "temp_macro_repo")

# URLs Reales de los repositorios científicos
REPOS = {
    "polidriving": "https://github.com/laboratorioAI/polidriving.git",
    # Usamos el repositorio Advanced-XLight que contiene los benchmarks clásicos
    "macro_benchmark": "https://github.com/wingsweihua/colight" 
}

def ensure_directories():
    """Crea la estructura de carpetas si no existe."""
    logging.info("Verificando estructura de directorios raw...")
    os.makedirs(POLIDRIVING_DIR, exist_ok=True)
    os.makedirs(MACRO_DIR, exist_ok=True)

def clone_polidriving():
    """Descarga el dataset microscópico latinoamericano."""
    logging.info("Iniciando ingesta de PoliDriving (Telemetría Ecuador)...")
    if not os.path.exists(os.path.join(POLIDRIVING_DIR, ".git")):
        subprocess.run(["git", "clone", REPOS["polidriving"], POLIDRIVING_DIR], check=True)
        logging.info("PoliDriving clonado con éxito.")
    else:
        logging.info("PoliDriving ya existe. Ejecutando git pull para actualizar...")
        subprocess.run(["git", "-C", POLIDRIVING_DIR, "pull"], check=True)

def download_macro_datasets():
    """
    Descarga los datasets de Hangzhou, Jinan y NYC clonando un repositorio 
    benchmark estándar y extrayendo las carpetas de datos útiles recursivamente.
    """
    logging.info("Iniciando ingesta de Datasets Macroscópicos...")
    
    # 1. Clonar el repositorio completo en una carpeta temporal
    if not os.path.exists(TEMP_REPO_DIR):
        logging.info("Clonando repositorio benchmark temporalmente...")
        subprocess.run(["git", "clone", REPOS["macro_benchmark"], TEMP_REPO_DIR], check=True)
    
    # 2. Buscar recursivamente las carpetas de las ciudades
    logging.info("Buscando redes de tráfico en el repositorio clonado...")
    found_datasets = False
    
    for root, dirs, files in os.walk(TEMP_REPO_DIR):
        # Evitar buscar dentro de la carpeta oculta .git
        if '.git' in root:
            continue
            
        for d in dirs:
            folder_name = d.lower()
            if "hangzhou" in folder_name or "jinan" in folder_name or "newyork" in folder_name:
                src_path = os.path.join(root, d)
                dest_path = os.path.join(MACRO_DIR, d)
                
                if not os.path.exists(dest_path):
                    logging.info(f"Extrayendo dataset macroscópico: {d}")
                    shutil.copytree(src_path, dest_path)
                    found_datasets = True
                else:
                    logging.info(f"El dataset {d} ya está en la carpeta macro_traffic.")
                    
    if not found_datasets:
        logging.warning("No se extrajeron nuevas carpetas. Verifica la estructura del repositorio.")

    # 3. Limpieza: Eliminar el repositorio pesado temporal
    logging.info("Limpiando archivos temporales...")
    if os.path.exists(TEMP_REPO_DIR):
        def handle_remove_readonly(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(TEMP_REPO_DIR, onerror=handle_remove_readonly)
        logging.info("Repositorio temporal eliminado.")

if __name__ == "__main__":
    logging.info("=== INICIANDO PIPELINE DE INGESTA DE DATOS ===")
    try:
        ensure_directories()
        clone_polidriving()
        download_macro_datasets()
        logging.info("=== INGESTA COMPLETADA EXITOSAMENTE ===")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error al ejecutar comando Git: {e}")
    except Exception as e:
        logging.error(f"Error inesperado en el pipeline: {e}")