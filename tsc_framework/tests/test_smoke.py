"""
test_smoke.py - Pruebas de Sanidad del Andamiaje
=================================================
Verifica que la estructura del proyecto y las importaciones base
funcionen correctamente antes de implementar lógica real.

Ejecutar con:
    pytest tests/ -v
"""

import importlib
import sys
from pathlib import Path

import pytest

# Asegurar que src/ esté en el path para importaciones
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TestPackageImports:
    """Valida que todos los módulos del paquete sean importables."""

    def test_import_data_pipeline(self):
        mod = importlib.import_module("data_pipeline")
        assert mod is not None

    def test_import_copulas(self):
        mod = importlib.import_module("copulas")
        assert mod is not None

    def test_import_rl_env(self):
        mod = importlib.import_module("rl_env")
        assert mod is not None

    def test_import_agents(self):
        mod = importlib.import_module("agents")
        assert mod is not None

    def test_import_utils(self):
        mod = importlib.import_module("utils")
        assert mod is not None


class TestProjectStructure:
    """Valida que los directorios y archivos esenciales existan."""

    def test_config_file_exists(self):
        config = ROOT / "config" / "default_config.yaml"
        assert config.exists(), f"No encontrado: {config}"

    def test_environment_yml_exists(self):
        env_file = ROOT / "environment.yml"
        assert env_file.exists(), f"No encontrado: {env_file}"

    def test_setup_py_exists(self):
        setup_file = ROOT / "setup.py"
        assert setup_file.exists(), f"No encontrado: {setup_file}"

    def test_data_dirs_exist(self):
        assert (ROOT / "data" / "raw").exists()
        assert (ROOT / "data" / "processed").exists()

    def test_sumo_dirs_exist(self):
        assert (ROOT / "sumo_configs" / "networks").exists()
        assert (ROOT / "sumo_configs" / "routes").exists()


class TestConfigLoading:
    """Valida que el YAML de configuración sea parseable."""

    def test_config_is_valid_yaml(self):
        import yaml
        config_path = ROOT / "config" / "default_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        import yaml
        config_path = ROOT / "config" / "default_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        required_keys = ["reproducibility", "sumo", "environment", "agent",
                         "risk_metrics", "copulas", "paths", "logging"]
        for key in required_keys:
            assert key in config, f"Clave faltante en config: '{key}'"

    def test_seed_is_integer(self):
        import yaml
        config_path = ROOT / "config" / "default_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert isinstance(config["reproducibility"]["global_seed"], int)
