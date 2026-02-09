"""Statistical tests and metrics for ICU simulation validation.

Classes
-------
Wilcoxon
    Paired Wilcoxon signed-rank test.
Friedman
    Friedman chi-square test for repeated measures.
SimulationMetrics
    Evaluation suite comparing real patient data against simulation output
    (coverage percentage, error margin, Kolmogorov–Smirnov, Anderson–Darling).
StatsUtils
    Static helpers (confidence intervals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeAlias, Union

import numpy as np
from scipy.stats import (
    anderson_ksamp,
    friedmanchisquare,
    ks_2samp,
    norm,
    t as t_dist,
    wilcoxon,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.constants import EXPERIMENT_VARIABLES_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Metric: TypeAlias = tuple[float, ...] | dict[str, Any]
ArrayLike1D = Union[Sequence[float], np.ndarray, Any]


# ---------------------------------------------------------------------------
# Simple hypothesis tests
# ---------------------------------------------------------------------------


@dataclass
class Wilcoxon:
    """Paired Wilcoxon signed-rank test wrapper"""

    x: ArrayLike1D
    y: ArrayLike1D

    statistic: float = field(init=False)
    p_value: float = field(init=False)

    def test(self) -> None:
        """Run the test and populate :attr:`statistic` and :attr:`p_value`."""

        res = wilcoxon(self.x, self.y)
        self.statistic = float(res.statistic)  # type: ignore[union-attr]
        self.p_value = float(res.pvalue)  # type: ignore[union-attr]


@dataclass
class Friedman:
    """Friedman chi-square test for *k* related samples."""

    samples: Sequence[Sequence[float]]

    statistic: float = 0.0
    p_value: float = 0.0

    def test(self) -> None:
        """Run the test and populate :attr:`statistic` and :attr:`p_value`"""

        res = friedmanchisquare(*self.samples)
        self.statistic = float(res.statistic)
        self.p_value = float(res.pvalue)


# ---------------------------------------------------------------------------
# Simulation evaluation metrics
# ---------------------------------------------------------------------------


@dataclass
class SimulationMetrics:
    """Compare real (true) patient data with simulation output.

    After calling :meth:`evaluate`, the result attributes
    (:attr:`coverage_percentage`, :attr:`error_margin`, etc.) are populated.
    """

    true_data: np.ndarray
    simulation_data: np.ndarray

    coverage_percentage: Metric | None = None
    error_margin: Metric | None = None
    kolmogorov_smirnov_result: Metric | None = None
    anderson_darling_result: Metric | None = None

    # ---- public API -------------------------------------------------------

    def evaluate(
        self,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        result_as_dict: bool = False,
    ) -> None:
        """Run all evaluation metrics.

        Args:
            confidence_level: Confidence level for coverage percentage (0.80–0.95 recommended).
            random_state: Seed for the RNG used in sampling-based tests. ``None`` for non-reproducible.
            result_as_dict: When ``True`` individual metrics return dicts instead of tuples.
        """

        np.random.seed(random_state)

        if not 0.80 <= confidence_level <= 0.95:
            logger.warning("Confidence level %.2f is outside the recommended 0.80–0.95 range", confidence_level)

        try:
            self.coverage_percentage = self._calculate_coverage_percentage(confidence_level=confidence_level)
            self.error_margin = self._calculate_error_margin(as_dict=result_as_dict)
            self.kolmogorov_smirnov_result = self._ks_test(as_dict=result_as_dict)
            self.anderson_darling_result = self._ad_test(as_dict=result_as_dict)
        except Exception:
            logger.exception("Evaluation failed")

    # ---- coverage percentage ----------------------------------------------

    def _calculate_coverage_percentage(self, confidence_level: float = 0.95) -> dict[str, float]:
        """Fraction of patients whose true value falls inside the simulation CI.

        Args:
            confidence_level: Confidence level for the t-based interval.

        Returns:
            Per-variable coverage percentages.
        """

        simulation_data = np.asarray(self.simulation_data)
        if simulation_data.ndim != 3:
            raise ValueError("simulation_data must be 3-D (n_patients, n_replicates, n_variables)")

        n_patients, n_replicates, n_variables = simulation_data.shape
        td = self._coerce_true_data(n_patients, n_variables)

        means = np.mean(simulation_data, axis=1)

        if n_replicates < 2:
            logger.warning("Fewer than 2 replicates — CI degenerates to the point estimate")
            lower = upper = means
        else:
            t_value = t_dist.ppf(1 - (1 - confidence_level) / 2, n_replicates - 1)
            sems = np.std(simulation_data, axis=1, ddof=1) / np.sqrt(n_replicates)
            margin = t_value * sems
            lower, upper = means - margin, means + margin

        in_ci = (td >= lower) & (td <= upper)
        coverage_array = in_ci.mean(axis=0) * 100

        return {
            self._variable_label(i): float(coverage_array[i])
            for i in range(n_variables)
        }

    # ---- error margin (RMSE / MAE / MAPE) ---------------------------------

    def _calculate_error_margin(self, as_dict: bool = False) -> Metric:
        """Compute RMSE, MAE and MAPE between true data and simulation means."""

        true_data = np.asarray(self.true_data)
        simulation_mean = np.mean(np.asarray(self.simulation_data), axis=1)

        td = self._align_shape(true_data, simulation_mean.shape)

        rmse = float(np.sqrt(mean_squared_error(td, simulation_mean)))
        mae = float(mean_absolute_error(td, simulation_mean))

        zero_mask = td == 0
        if np.all(zero_mask):
            mape = float("nan")
            logger.warning("MAPE undefined — all true values are zero")
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.abs((td - simulation_mean) / td.astype(float))
                mape = float(np.nanmean(np.where(zero_mask, np.nan, raw)) * 100)

        logger.info("RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%", rmse, mae, mape)

        if as_dict:
            return {"rmse": rmse, "mae": mae, "mape": mape}
        return (rmse, mae, mape)

    # ---- Kolmogorov–Smirnov -----------------------------------------------

    def _ks_test(self, as_dict: bool = False) -> Metric:
        """Two-sample Kolmogorov–Smirnov test (per-variable when data is 3-D)."""

        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        if simulation_data.ndim == 3:
            n_vars = simulation_data.shape[2]
            per_var: list[tuple[float, float]] = []

            for v in range(n_vars):
                sim_flat = simulation_data[:, :, v].ravel()
                true_flat = true_data[:, v].ravel() if true_data.ndim > 1 else true_data.ravel()
                try:
                    ks_res = ks_2samp(true_flat, sim_flat)
                    stat, p = float(ks_res.statistic), float(ks_res.pvalue)  # type: ignore[union-attr]
                except Exception:
                    stat, p = float("nan"), float("nan")
                per_var.append((stat, p))

            stats_arr = np.array([s for s, _ in per_var])
            pvals_arr = np.array([p for _, p in per_var])
            overall_stat = float(np.nanmean(stats_arr))
            overall_p = float(np.nanmean(pvals_arr))

            logger.info("KS per-variable: %s", per_var)
            logger.info("KS overall — stat=%.4f  p=%.4f", overall_stat, overall_p)

            if as_dict:
                per_variable_dict = {
                    self._variable_label(i): {"statistic": s, "p_value": p}
                    for i, (s, p) in enumerate(per_var)
                }
                return {"per_variable": per_variable_dict, "overall": {"statistic": overall_stat, "p_value": overall_p}}
            return (overall_stat, overall_p)
        else:
            try:
                ks_res = ks_2samp(true_data, simulation_data.flatten())
                stat, p = float(ks_res.statistic), float(ks_res.pvalue)  # type: ignore[union-attr]
            except Exception:
                stat, p = float("nan"), float("nan")

            logger.info("KS — stat=%.4f  p=%.4f", stat, p)
            if as_dict:
                return {"statistic": stat, "p_value": p}
            return (stat, p)

    # ---- Anderson–Darling -------------------------------------------------

    def _ad_test(self, as_dict: bool = False) -> Metric:
        """Anderson–Darling *k*-sample test comparing true vs. simulated data."""
        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        if simulation_data.ndim == 3:
            sim_mean = np.mean(simulation_data, axis=1)
            min_size = min(true_data.size, sim_mean.size)
            real_sample = np.random.choice(true_data.flatten(), min_size, replace=False)
            simulated_sample = np.random.choice(sim_mean.flatten(), min_size, replace=False)
        else:
            min_size = min(len(true_data), len(simulation_data))
            real_sample = np.random.choice(true_data, min_size, replace=False)
            simulated_sample = np.random.choice(simulation_data, min_size, replace=False)

        try:
            try:
                from scipy.stats import PermutationMethod
                result = anderson_ksamp([real_sample, simulated_sample], method=PermutationMethod())
            except ImportError:
                result = anderson_ksamp([real_sample, simulated_sample])
            statistic = float(result.statistic)  # type: ignore[union-attr]
            significance = float(getattr(result, "significance_level", float("nan")))
        except Exception:
            logger.exception("Anderson–Darling test failed")
            statistic = float("nan")
            significance = float("nan")

        logger.info("Anderson-Darling — stat=%.4f  p≈%.3f", statistic, significance)

        if as_dict:
            return {"statistic": statistic, "significance_level": significance}
        return (statistic, significance)

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _variable_label(index: int) -> str:
        """Return a human-readable variable name for *index*."""
        if index < len(EXPERIMENT_VARIABLES_LABELS):
            return EXPERIMENT_VARIABLES_LABELS[index]
        return f"variable_{index}"

    @staticmethod
    def _align_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        """Best-effort reshape / resize of *arr* to *target_shape*."""
        if arr.shape == target_shape:
            return arr
        if arr.size == np.prod(target_shape):
            return arr.reshape(target_shape)
        return np.resize(arr, target_shape)

    def _coerce_true_data(self, n_patients: int, n_variables: int) -> np.ndarray:
        """Coerce :attr:`true_data` into shape ``(n_patients, n_variables)``.

        Handles scalar, 1-D, 2-D and higher-dimensional inputs gracefully,
        logging a warning whenever an implicit broadcast or trim is applied.
        """
        td = np.asarray(self.true_data)
        target = (n_patients, n_variables)

        if td.ndim == 0:
            return np.full(target, float(td))

        if td.ndim == 1:
            if td.size == n_patients * n_variables:
                return td.reshape(target)
            if td.size == n_variables:
                return np.tile(td.reshape(1, n_variables), (n_patients, 1))
            if td.size == n_patients:
                if n_variables == 1:
                    return td.reshape(n_patients, 1)
                logger.warning("true_data has length n_patients; tiling across variables")
                return np.tile(td.reshape(n_patients, 1), (1, n_variables))
            logger.warning("true_data length mismatch; resizing to (%d, %d)", *target)
            return np.resize(td, target)

        if td.ndim == 2:
            rows, cols = td.shape
            if rows >= n_patients and cols >= n_variables:
                return td[:n_patients, :n_variables]
            logger.warning("true_data smaller than expected; resizing to (%d, %d)", *target)
            return np.resize(td, target)

        logger.warning("true_data has %d dimensions; flattening and resizing", td.ndim)
        return np.resize(td.ravel(), target)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class StatsUtils:
    """Static statistical helper methods."""

    @staticmethod
    def confidence_interval(
        mean: ArrayLike1D,
        std: ArrayLike1D,
        n: int,
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a confidence interval for the given summary statistics.

        Args:
            mean: Sample means.
            std: Sample standard deviations.
            n: Sample size.
            confidence: Confidence level (default 0.95).

        Returns:
            ``(lower_bound, upper_bound)`` arrays.
        """
        mean = np.asarray(mean)
        std = np.asarray(std)

        if np.all(std == 0):
            return mean, mean

        t_value = t_dist.ppf(1 - (1 - confidence) / 2, n - 1)
        sem = std / np.sqrt(n)

        return mean - t_value * sem, mean + t_value * sem
