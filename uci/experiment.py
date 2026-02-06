"""Experiment definition and execution for ICU simulation.

Provides the :class:`Experiment` data holder and the runner functions
:func:`single_run` and :func:`multiple_replication` that drive the
discrete-event simulation.
"""

from __future__ import annotations

import logging

import pandas as pd
import simpy

from uci import distribuciones
from uci.simulacion import Simulation
from utils.constants import EXPERIMENT_VARIABLES_LABELS

logger = logging.getLogger(__name__)


class Experiment:
    """Patient-level parameters and mutable result container for one simulation run."""

    def __init__(
        self,
        age: int,
        diagnosis_admission1: int,
        diagnosis_admission2: int,
        diagnosis_admission3: int,
        diagnosis_admission4: int,
        apache: int,
        respiratory_insufficiency: int,
        artificial_ventilation: int,
        uti_stay: int,
        vam_time: int,
        preuti_stay_time: int,
        percent: int = 10,
    ) -> None:
        self.edad: int = age
        self.diagn1: int = diagnosis_admission1
        self.diagn2: int = diagnosis_admission2
        self.diagn3: int = diagnosis_admission3
        self.diagn4: int = diagnosis_admission4
        self.apache: int = apache
        self.insuf_resp: int = respiratory_insufficiency
        self.va: int = artificial_ventilation
        self.estadia_uti: int = uti_stay
        self.tiempo_vam: int = vam_time
        self.tiempo_pre_uti: int = preuti_stay_time
        self.porciento: int = percent

        self.result: dict[str, int] = {}

    def init_results_variables(self) -> None:
        """Reset result dict with zeroes for every experiment variable."""

        self.result = {var: 0 for var in EXPERIMENT_VARIABLES_LABELS}


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


def single_run(experiment: Experiment) -> dict[str, int]:
    """Execute one simulation replication and return the result dict."""

    env = simpy.Environment()
    experiment.init_results_variables()

    cluster = distribuciones.clustering(
        experiment.edad,
        experiment.diagn1,
        experiment.diagn2,
        experiment.diagn3,
        experiment.diagn4,
        experiment.apache,
        experiment.insuf_resp,
        experiment.va,
        experiment.estadia_uti,
        experiment.tiempo_vam,
        experiment.tiempo_pre_uti,
    )

    simulation = Simulation(experiment, cluster)
    env.process(simulation.uci(env))
    env.run()

    return experiment.result


def multiple_replication(
    experiment: Experiment,
    n_reps: int = 100,
    as_int: bool = True,
) -> pd.DataFrame:
    """Run *n_reps* independent replications and return results as a DataFrame.

    Args:
        experiment: Configured :class:`Experiment` instance.
        n_reps: Number of independent replications.
        as_int: If ``True`` cast every value to ``int64``; otherwise keep ``float64``.

    Returns:
        A :class:`~pandas.DataFrame` with one row per replication.
    """
    results: list[dict[str, int | float]] = []

    for _ in range(n_reps):
        raw = single_run(experiment)

        row: dict[str, int | float] = {}
        for key, value in raw.items():
            try:
                val = float(value)
            except (ValueError, TypeError):
                val = 0.0
            row[key] = int(val) if as_int else val
        results.append(row)

    df = pd.DataFrame(results)

    # Fill any unexpected NaN and enforce column types
    df = df.fillna(0 if as_int else 0.0)
    target_dtype = "int64" if as_int else "float64"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(target_dtype)

    return df
