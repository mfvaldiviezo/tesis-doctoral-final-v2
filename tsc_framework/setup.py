"""
setup.py - Instalación local del paquete tsc_framework
======================================================
Permite instalar el directorio src/ como un paquete Python editable
mediante:  pip install -e .

Esto habilita importaciones directas como:
    from src.rl_env.sumo_env import SumoEnv
    from src.agents.ppo_agent import PPOAgent
    from src.copulas.vine_generator import VineScenarioGenerator
"""

from setuptools import setup, find_packages

# Dependencias mínimas (el entorno completo se gestiona vía environment.yml)
INSTALL_REQUIRES = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "gymnasium>=0.29",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "rich>=13.0",
    "python-dotenv>=1.0",
]

EXTRAS_REQUIRE = {
    "rl": [
        "stable-baselines3>=2.2.1",
        "tensorboard>=2.14",
    ],
    "copulas": [
        "pyvinecopulib>=0.6.3",
    ],
    "explain": [
        "shap>=0.44",
    ],
    "dev": [
        "pytest>=7.4",
        "pytest-cov>=4.1",
        "jupyterlab>=4.0",
    ],
    "all": [
        "stable-baselines3>=2.2.1",
        "tensorboard>=2.14",
        "pyvinecopulib>=0.6.3",
        "shap>=0.44",
        "pytest>=7.4",
        "pytest-cov>=4.1",
    ],
}

setup(
    name="tsc_framework",
    version="0.1.0",
    description=(
        "Framework modular para control semafórico inteligente basado en "
        "Aprendizaje por Refuerzo sensible al riesgo, Vine Copulas y métricas "
        "de equidad distributiva. Tesis Doctoral - IA Aplicada."
    ),
    author="Doctoral Researcher",
    author_email="",
    python_requires=">=3.10",
    # Busca todos los paquetes dentro de src/
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    # Registrar puntos de entrada como comandos de consola (opcional)
    entry_points={
        "console_scripts": [
            "tsc-train=scripts.train:main",
            "tsc-evaluate=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
)
