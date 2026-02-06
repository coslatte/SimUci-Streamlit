"""Data loading and column extraction utilities for patient CSV files.

All public helpers accept a *path* to a CSV file and return either a
sorted column (as a list) or a generator that yields values one by one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import pandas as pd

logger = logging.getLogger(__name__)

# Column names expected in the patient CSV
_DATE_COLUMNS = ["fecha_ingreso", "fecha_egreso", "fecha_ing_uci", "fecha_egr_uci"]


def cargar_fichero(path: str | Path, column: str) -> list:
    """Load a patient CSV and return one column sorted by admission date.

    Args:
        path: Path to the CSV data file.
        column: Name of the column to extract.

    Returns:
        A list with the values of *column*, ordered by ``fecha_ingreso``.
    """
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=_DATE_COLUMNS,
    )
    df["tiempo_vam"] = df["tiempo_vam"].astype(int)
    df["diagnostico_preuci"] = df["diagnostico_preuci"].astype("category")
    return list(df.sort_values("fecha_ingreso")[column])


# ---------------------------------------------------------------------------
# Column generators
# ---------------------------------------------------------------------------

def _iter_column(path: str | Path, column: str) -> Generator:
    """Yield every value from *column* (sorted by admission date)."""
    yield from cargar_fichero(path, column)


def get_fecha_ingreso(path: str | Path) -> Generator[tuple, None, None]:
    """Yield ``(next_date, current_date)`` tuples for admission dates."""
    fechas = cargar_fichero(path, "fecha_ingreso")
    current = fechas[0]
    for siguiente in fechas:
        yield (siguiente, current)
        current = siguiente


def get_fecha_egreso(path: str | Path) -> Generator:
    """Yield discharge dates."""
    yield from _iter_column(path, "fecha_egreso")


def get_fecha_ing_uci(path: str | Path) -> Generator:
    """Yield ICU admission dates."""
    yield from _iter_column(path, "fecha_ing_uci")


def get_tiempo_vam(path: str | Path) -> Generator:
    """Yield VAM time (hours) for each patient."""
    yield from _iter_column(path, "tiempo_vam")


def get_fecha_egr_uci(path: str | Path) -> Generator:
    """Yield ICU discharge dates."""
    yield from _iter_column(path, "fecha_egr_uci")


def get_estadia_uci(path: str | Path) -> Generator:
    """Yield ICU stay duration for each patient."""
    yield from _iter_column(path, "estadia_uci")


def get_sala_egreso(path: str | Path) -> Generator:
    """Yield discharge ward for each patient."""
    yield from _iter_column(path, "sala_egreso")


def get_evolucion(path: str | Path) -> Generator:
    """Yield patient outcome (survived / deceased)."""
    yield from _iter_column(path, "evolucion")


def get_diagnostico(path: str | Path) -> Generator:
    """Yield pre-ICU diagnosis for each patient."""
    yield from _iter_column(path, "diagnostico_preuci")


# ---------------------------------------------------------------------------
# Aggregate queries
# ---------------------------------------------------------------------------

def get_diagnostico_list(path: str | Path) -> list:
    """Return the unique pre-ICU diagnoses present in the data file.

    Args:
        path: Path to the CSV data file.

    Returns:
        Array-like of unique diagnosis values.
    """
    df = pd.read_csv(path, index_col=0)
    return list(df["diagnostico_preuci"].unique())


def get_time_simulation(path: str | Path) -> int:
    """Compute total simulation horizon in hours from admission to last discharge.

    Args:
        path: Path to the CSV data file.

    Returns:
        Number of hours between the earliest admission and the latest discharge.
    """
    first_admission = cargar_fichero(path, "fecha_ingreso")[0]
    last_discharge = max(cargar_fichero(path, "fecha_egreso"))
    span = last_discharge - first_admission
    hours: int = span.days * 24
    logger.debug("Simulation horizon: %s (%d hours)", span, hours)
    return hours
