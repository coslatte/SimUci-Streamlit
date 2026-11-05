"""
framework
=========
Marco de trabajo para la evaluación de incertidumbre y robustez del
modelo SimUci.

Módulos
-------
uncertainty_module   — Bootstrapping, Conformal prediction, Monte Carlo
robustness_module    — Stress-tests, índice R, semáforo clínico
simuci_adapter       — Puente con el núcleo real de SimUci
dashboard_integration — Componentes Streamlit del dashboard
"""

from .uncertainty_module import UncertaintyQuantifier, PredictionInterval
from .robustness_module import RobustnessEvaluator, RobustnessReport
from .simuci_adapter import SimUciAdapter

__all__ = [
    "UncertaintyQuantifier",
    "PredictionInterval",
    "RobustnessEvaluator",
    "RobustnessReport",
    "SimUciAdapter",
]
__version__ = "1.0.0"
