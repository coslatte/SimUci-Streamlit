"""
simuci_framework
=================
Marco de trabajo para la evaluación de incertidumbre y robustez del
modelo SimUci. Diseñado para copiarse íntegramente a la raíz del proyecto
SimUci-Streamlit como un subpaquete independiente, sin modificar ningún
archivo existente del proyecto (integración no invasiva, RNF-03).

Uso desde app.py (raíz del proyecto SimUci)
--------------------------------------------
    from simuci_framework import (
        UncertaintyQuantifier, RobustnessEvaluator, SimUciAdapter,
    )
    from simuci_framework.framework.dashboard_integration import FrameworkDashboard

    fw = FrameworkDashboard()
    fw.render()

Autor   : Jorge Jesús Santos García
Tutora  : DrC. Deborah Raquel Galpert Cañizares
Centro  : UCLV — Departamento de Ciencias de la Computación
Año     : 2025
"""

from .framework import (
    UncertaintyQuantifier,
    PredictionInterval,
    RobustnessEvaluator,
    RobustnessReport,
    SimUciAdapter,
)

__all__ = [
    "UncertaintyQuantifier",
    "PredictionInterval",
    "RobustnessEvaluator",
    "RobustnessReport",
    "SimUciAdapter",
]
__version__ = "1.0.0"
