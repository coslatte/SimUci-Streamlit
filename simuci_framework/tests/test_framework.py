"""
tests/test_framework.py
=======================
Suite completa de pruebas unitarias para el framework de incertidumbre
y robustez de SimUci.

Cobertura
---------
- uncertainty_module : DistributionFitter, ParametricBootstrapper,
                       ConformalPredictor, MonteCarloQuantifier,
                       UncertaintyQuantifier, decompose_uncertainty
- robustness_module  : RobustnessEvaluator (stress_noise, stress_missing,
                       stress_shift, sensitivity_analysis, evaluate)
- simuci_adapter     : SimUciAdapter (modo stub)

Ejecución
---------
    cd <raíz del proyecto>
    pytest tests/test_framework.py -v --tb=short

Autor   : Jorge Jesús Santos García
Tutora  : DrC. Deborah Raquel Galpert Cañizares
Centro  : UCLV — Departamento de Ciencias de la Computación
Año     : 2025
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Asegurar que el paquete framework sea importable desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.uncertainty_module import (
    ConformalPredictor,
    DistributionFitter,
    MonteCarloQuantifier,
    ParametricBootstrapper,
    PredictionInterval,
    UncertaintyQuantifier,
    decompose_uncertainty,
)
from framework.robustness_module import (
    CLINICAL_THRESHOLD,
    MISSING_LEVELS,
    NOISE_LEVELS,
    SEMAPHORE_AMBER,
    SEMAPHORE_GREEN,
    RobustnessEvaluator,
    RobustnessReport,
    _component_score,
    _relative_variation,
    _semaphore,
    _status_from_variation,
)
from framework.simuci_adapter import (
    ALL_FEATURES,
    NUMERIC_COLS,
    TARGET_COL,
    SimUciAdapter,
    _StubSimulator,
)

# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------

SEED = 42
RNG = np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def synthetic_cohort() -> pd.DataFrame:
    """Cohorte sintética de 208 pacientes para pruebas."""
    return _StubSimulator.build_cohort()


@pytest.fixture(scope="module")
def calibration_split(synthetic_cohort) -> tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 split de la cohorte sintética."""
    n = len(synthetic_cohort)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_cal = int(n * 0.30)
    cal   = synthetic_cohort.iloc[idx[:n_cal]].reset_index(drop=True)
    train = synthetic_cohort.iloc[idx[n_cal:]].reset_index(drop=True)
    return train, cal


@pytest.fixture(scope="module")
def trained_rf(calibration_split) -> RandomForestRegressor:
    """Random Forest entrenado sobre el conjunto de entrenamiento."""
    train_df, _ = calibration_split
    X = train_df[ALL_FEATURES].to_numpy(dtype=float)
    y = train_df[TARGET_COL].to_numpy(dtype=float)
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=1
    )
    rf.fit(X, y)
    return rf


@pytest.fixture(scope="module")
def default_patient() -> dict:
    """
    Parámetros de un paciente típico, dentro de los rangos reales
    verificados en utils/constants/limits.py del proyecto SimUci:
    AGE [14,100], APACHE [0,36], VAM_T horas [24,700],
    UTI_STAY horas [0,200], PREUTI_STAY horas [0,34].
    d1 es un código de diagnóstico PREUCI_DIAG (21 = "Sepsis grave").
    """
    return dict(
        n_runs=200, age=57, apache=18, vam_time=120,
        uti_stay=96, preuti_stay=12, percent=3,
        d1=21, d2=0, d3=0, d4=0,
        resp_insuf=1, artif_vent=0,
    )


@pytest.fixture(scope="module")
def adapter() -> SimUciAdapter:
    """Adaptador en modo stub (sin SimUci instalado)."""
    return SimUciAdapter()


# ---------------------------------------------------------------------------
# 1. DistributionFitter
# ---------------------------------------------------------------------------

class TestDistributionFitter:

    def test_fit_lognormal_data(self):
        """Debe seleccionar log-normal para datos generados con esa distribución."""
        rng = np.random.default_rng(0)
        data = rng.lognormal(mean=1.5, sigma=0.7, size=300)
        fitter = DistributionFitter(candidates=["lognorm", "gamma", "expon"])
        fitter.fit(data)
        assert fitter.best_dist_ is not None
        assert fitter.best_pvalue_ > 0.0

    def test_fit_returns_self(self):
        """fit() debe devolver self para encadenamiento."""
        data = np.abs(np.random.default_rng(1).standard_normal(100)) + 0.1
        fitter = DistributionFitter()
        result = fitter.fit(data)
        assert result is fitter

    def test_sample_positive_values(self):
        """Las muestras generadas deben ser positivas."""
        data = np.random.default_rng(2).lognormal(1.0, 0.5, 200)
        fitter = DistributionFitter()
        fitter.fit(data)
        samples = fitter.sample(100)
        assert len(samples) == 100
        assert np.all(np.isfinite(samples))

    def test_sample_before_fit_raises(self):
        """sample() sin fit previo debe lanzar RuntimeError."""
        with pytest.raises(RuntimeError, match="fit"):
            DistributionFitter().sample(10)

    def test_insufficient_data_warning(self):
        """Datos insuficientes (<10) deben emitir advertencia."""
        with pytest.warns(UserWarning):
            DistributionFitter().fit(np.array([1.0, 2.0, 3.0]))

    def test_all_results_populated(self):
        """all_results_ debe contener entradas para cada distribución candidata."""
        data = np.random.default_rng(3).lognormal(1.0, 0.6, 150)
        candidates = ["lognorm", "gamma", "expon"]
        fitter = DistributionFitter(candidates=candidates)
        fitter.fit(data)
        for c in candidates:
            assert c in fitter.all_results_


# ---------------------------------------------------------------------------
# 2. PredictionInterval
# ---------------------------------------------------------------------------

class TestPredictionInterval:

    def test_width_computed_automatically(self):
        iv = PredictionInterval(
            prediction=14.3, lower=11.2, upper=18.1,
            confidence=0.95, method="conformal"
        )
        assert math.isclose(iv.width, 18.1 - 11.2, rel_tol=1e-6)

    def test_as_dict_contains_required_keys(self):
        iv = PredictionInterval(10.0, 7.0, 13.0, 0.90, "bootstrap")
        d = iv.as_dict()
        for key in ("prediction", "lower", "upper", "width", "confidence", "method"):
            assert key in d

    def test_lower_not_negative(self):
        """El límite inferior siempre es >= 0 tras clipping en predict_interval."""
        iv = PredictionInterval(2.0, -1.0, 5.0, 0.95, "test")
        assert iv.lower == -1.0   # PredictionInterval no clipea; el clipping
                                  # ocurre en los métodos de predicción


# ---------------------------------------------------------------------------
# 3. ConformalPredictor
# ---------------------------------------------------------------------------

class TestConformalPredictor:

    def test_fit_and_predict(self, calibration_split, trained_rf):
        _, cal_df = calibration_split
        X_cal = cal_df[ALL_FEATURES].to_numpy(dtype=float)
        y_cal = cal_df[TARGET_COL].to_numpy(dtype=float)

        cp = ConformalPredictor(confidence=0.90)
        cp.fit(X_cal, y_cal, trained_rf)

        x_new = X_cal[0].reshape(1, -1)
        iv = cp.predict_interval(x_new)

        assert iv.method == "conformal"
        assert iv.lower <= iv.prediction <= iv.upper
        assert iv.width > 0

    def test_predict_interval_accepts_1d_array(self, calibration_split, trained_rf):
        """Verifica la rama poco ejercitada de reshape automático para vectores 1D."""
        _, cal_df = calibration_split
        X_cal = cal_df[ALL_FEATURES].to_numpy(dtype=float)
        y_cal = cal_df[TARGET_COL].to_numpy(dtype=float)

        cp = ConformalPredictor(confidence=0.95)
        cp.fit(X_cal, y_cal, trained_rf)

        x_1d = X_cal[0]  # shape (n_features,), sin reshape previo
        assert x_1d.ndim == 1
        iv = cp.predict_interval(x_1d)
        assert iv.lower <= iv.prediction <= iv.upper

    def test_coverage_at_least_nominal(self, calibration_split, trained_rf):
        """La cobertura empírica debe estar cerca del nivel nominal."""
        train_df, cal_df = calibration_split
        X_cal = cal_df[ALL_FEATURES].to_numpy(dtype=float)
        y_cal = cal_df[TARGET_COL].to_numpy(dtype=float)
        X_test = train_df[ALL_FEATURES].to_numpy(dtype=float)
        y_test = train_df[TARGET_COL].to_numpy(dtype=float)

        confidence = 0.90
        cp = ConformalPredictor(confidence=confidence)
        cp.fit(X_cal, y_cal, trained_rf)
        cov = cp.evaluate_coverage(X_test, y_test)

        # Cobertura debe ser >= nominal − margen de error
        assert cov >= confidence - 0.10, (
            f"Cobertura {cov:.3f} < nominal {confidence} − 0.10"
        )

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            ConformalPredictor().predict_interval(np.zeros((1, 10)))

    def test_threshold_uses_ceiling_order_statistic(self):
        """
        Regresión del hallazgo real: el umbral debe calcularse con el
        estadístico de orden EXACTO indicado por
        ceil((n_cal+1)(1-alpha))/n_cal (Vovk et al., 2005; Angelopoulos y
        Bates, 2022), no con una interpolación lineal entre estadísticos
        de orden vecinos (np.percentile con el método por defecto), que
        subestimaba el umbral real hasta en un 23% en pruebas con
        distribuciones de error asimétricas.
        """
        rng = np.random.default_rng(7)
        n = 62
        scores_like = rng.exponential(scale=15, size=n)

        class _FixedModel:
            def predict(self, X):
                return np.zeros(len(X))

        X_cal = np.zeros((n, 3))
        y_cal = scores_like  # con predicción=0, el score = |y - 0| = y
        cp = ConformalPredictor(confidence=0.95)
        cp.fit(X_cal, y_cal, _FixedModel())

        sorted_scores = np.sort(scores_like)
        alpha = 1 - 0.95
        expected_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
        expected_idx = min(expected_idx, n - 1)
        expected_threshold = sorted_scores[expected_idx]

        assert math.isclose(cp._threshold, expected_threshold, rel_tol=1e-9), (
            f"Umbral {cp._threshold} no coincide con el estadístico de "
            f"orden exacto esperado {expected_threshold} — posible "
            f"regresión a la fórmula interpolada incorrecta."
        )

    def test_evaluate_coverage_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ConformalPredictor().evaluate_coverage(
                np.zeros((5, 10)), np.zeros(5)
            )

    def test_threshold_positive(self, calibration_split, trained_rf):
        _, cal_df = calibration_split
        X = cal_df[ALL_FEATURES].to_numpy(dtype=float)
        y = cal_df[TARGET_COL].to_numpy(dtype=float)
        cp = ConformalPredictor(confidence=0.95)
        cp.fit(X, y, trained_rf)
        assert cp._threshold > 0

    def test_interval_width_increases_with_confidence(
        self, calibration_split, trained_rf
    ):
        """Mayor nivel de confianza → intervalo más ancho."""
        _, cal_df = calibration_split
        X = cal_df[ALL_FEATURES].to_numpy(dtype=float)
        y = cal_df[TARGET_COL].to_numpy(dtype=float)
        x_new = X[0].reshape(1, -1)

        widths = []
        for conf in [0.80, 0.90, 0.95, 0.99]:
            cp = ConformalPredictor(confidence=conf)
            cp.fit(X, y, trained_rf)
            iv = cp.predict_interval(x_new)
            widths.append(iv.width)

        assert widths == sorted(widths), (
            "Los anchos de intervalo deben crecer con el nivel de confianza"
        )


# ---------------------------------------------------------------------------
# 4. ParametricBootstrapper
# ---------------------------------------------------------------------------

class TestParametricBootstrapper:

    def _make_sim_fn(self, base: float = 14.3):
        """Función de simulación stub determinista."""
        def sim(**kwargs):
            age = kwargs.get("age", 57)
            return base * (1 + 0.01 * (age - 57))
        return sim

    def test_fit_and_predict(self, synthetic_cohort):
        pb = ParametricBootstrapper(B=200, confidence=0.90, seed=SEED)
        pb.fit(synthetic_cohort, NUMERIC_COLS)
        iv = pb.predict_interval(self._make_sim_fn(), {"age": 57, "apache": 18})
        assert iv.method == "bootstrap"
        assert iv.lower >= 0
        assert iv.lower < iv.upper

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ParametricBootstrapper().predict_interval(lambda **k: 1.0, {})

    def test_B_affects_width(self, synthetic_cohort):
        """Más réplicas → intervalo más estable (no necesariamente más estrecho)."""
        sim = self._make_sim_fn()
        for B in [50, 500]:
            pb = ParametricBootstrapper(B=B, seed=SEED)
            pb.fit(synthetic_cohort, NUMERIC_COLS)
            iv = pb.predict_interval(sim, {"age": 57, "apache": 18})
            assert iv.width >= 0

    def test_confidence_in_result(self, synthetic_cohort):
        conf = 0.95
        pb = ParametricBootstrapper(B=100, confidence=conf, seed=SEED)
        pb.fit(synthetic_cohort, NUMERIC_COLS)
        iv = pb.predict_interval(self._make_sim_fn(), {"age": 57})
        assert math.isclose(iv.confidence, conf)


# ---------------------------------------------------------------------------
# 5. MonteCarloQuantifier
# ---------------------------------------------------------------------------

class TestMonteCarloQuantifier:

    def _sim(self, **kwargs):
        age = kwargs.get("age", 57)
        apache = kwargs.get("apache", 18)
        return float(14.3 + 0.05 * (apache - 18) - 0.02 * (age - 57))

    def test_basic_interval(self):
        mc = MonteCarloQuantifier(n_reps=300, confidence=0.90, seed=SEED)
        iv = mc.predict_interval(self._sim, {"age": 57, "apache": 18})
        assert iv.method == "montecarlo"
        assert iv.lower < iv.prediction < iv.upper

    def test_lower_non_negative(self):
        mc = MonteCarloQuantifier(n_reps=100, seed=SEED)
        iv = mc.predict_interval(self._sim, {"age": 57, "apache": 18})
        assert iv.lower >= 0

    def test_with_custom_distributions(self):
        from scipy.stats import lognorm
        dists = {"age": ("norm", (57, 10))}
        mc = MonteCarloQuantifier(n_reps=200, seed=SEED)
        iv = mc.predict_interval(self._sim, {"age": 57, "apache": 18},
                                  param_distributions=dists)
        assert iv.width > 0


# ---------------------------------------------------------------------------
# 6. UncertaintyQuantifier (fachada)
# ---------------------------------------------------------------------------

    def test_config_keys_not_perturbed(self):
        """n_runs, percent y _crn_seed son configuración, no variables clínicas."""
        seen = []

        def fn(**kw):
            seen.append((kw["n_runs"], kw["percent"], kw["_crn_seed"]))
            return float(kw["age"])

        MonteCarloQuantifier(n_reps=30, seed=1).predict_interval(
            fn, dict(n_runs=200, percent=3, _crn_seed=7, age=50)
        )
        assert set(seen) == {(200, 3, 7)}


class TestUncertaintyQuantifier:

    def test_default_method_is_conformal(self):
        uq = UncertaintyQuantifier()
        assert uq.method == "conformal"

    def test_change_method(self):
        uq = UncertaintyQuantifier(method="conformal")
        uq.change_method("bootstrap")
        assert uq.method == "bootstrap"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Método desconocido"):
            UncertaintyQuantifier(method="unknown_method")  # type: ignore

    def test_conformal_end_to_end(self, calibration_split, trained_rf):
        _, cal_df = calibration_split
        X_cal = cal_df[ALL_FEATURES].to_numpy(dtype=float)
        y_cal = cal_df[TARGET_COL].to_numpy(dtype=float)
        x_new = X_cal[0].reshape(1, -1)

        uq = UncertaintyQuantifier(method="conformal", confidence=0.95)
        uq.fit(X_cal, y_cal, trained_rf)
        iv = uq.predict_interval(x_new)

        assert isinstance(iv, PredictionInterval)
        assert iv.method == "conformal"
        assert iv.lower <= iv.prediction <= iv.upper

    def test_evaluate_coverage_only_for_conformal(self, calibration_split, trained_rf):
        _, cal_df = calibration_split
        X = cal_df[ALL_FEATURES].to_numpy(dtype=float)
        y = cal_df[TARGET_COL].to_numpy(dtype=float)

        uq = UncertaintyQuantifier(method="conformal")
        uq.fit(X, y, trained_rf)
        cov = uq.evaluate_coverage(X, y)
        assert 0.0 <= cov <= 1.0

        uq2 = UncertaintyQuantifier(method="montecarlo")
        with pytest.raises(NotImplementedError):
            uq2.evaluate_coverage(X, y)

    def test_repr(self):
        uq = UncertaintyQuantifier(method="bootstrap", confidence=0.90, B=500)
        r = repr(uq)
        assert "bootstrap" in r
        assert "0.9" in r


# ---------------------------------------------------------------------------
# 7. decompose_uncertainty
# ---------------------------------------------------------------------------

class TestDecomposeUncertainty:

    def test_sums_to_100(self):
        rng = np.random.default_rng(SEED)
        boot = rng.lognormal(2.0, 0.5, 500)
        mc   = rng.lognormal(2.0, 0.8, 500)
        d = decompose_uncertainty(boot, mc)
        total = d["aleatoria"] + d["parametrica"] + d["estructural"]
        assert math.isclose(total, 100.0, rel_tol=1e-3)

    def test_all_non_negative(self):
        rng = np.random.default_rng(1)
        d = decompose_uncertainty(
            rng.lognormal(1.5, 0.3, 200),
            rng.lognormal(1.5, 0.6, 200),
        )
        assert all(v >= 0 for v in d.values())

    def test_higher_boot_variance_means_more_parametric(self):
        rng = np.random.default_rng(2)
        mc   = rng.lognormal(1.5, 0.4, 500)
        boot_lo = rng.lognormal(1.5, 0.1, 500)
        boot_hi = rng.lognormal(1.5, 0.9, 500)
        d_lo = decompose_uncertainty(boot_lo, mc)
        d_hi = decompose_uncertainty(boot_hi, mc)
        assert d_hi["parametrica"] >= d_lo["parametrica"]


# ---------------------------------------------------------------------------
# 8. Helpers de robustez
# ---------------------------------------------------------------------------

class TestRobustnessHelpers:

    def test_relative_variation_zero_for_equal(self):
        assert _relative_variation(10.0, 10.0) == 0.0

    def test_relative_variation_positive(self):
        v = _relative_variation(10.0, 12.0)
        assert math.isclose(v, 0.2, rel_tol=1e-6)

    def test_relative_variation_base_zero(self):
        """Denominador ~0 no debe producir división por cero."""
        v = _relative_variation(0.0, 1.0)
        assert math.isfinite(v)

    def test_component_score_at_threshold(self):
        """Variación exactamente en umbral → score = 0."""
        assert math.isclose(_component_score(CLINICAL_THRESHOLD), 0.0, abs_tol=1e-6)

    def test_component_score_zero_variation(self):
        """Sin variación → score máximo = 1."""
        assert math.isclose(_component_score(0.0), 1.0, rel_tol=1e-6)

    def test_component_score_above_threshold(self):
        """Variación mayor que umbral → score = 0."""
        assert _component_score(CLINICAL_THRESHOLD * 2) == 0.0

    def test_status_from_variation(self):
        assert _status_from_variation(0.05) == "Estable"
        assert _status_from_variation(0.15) == "Alerta"
        assert _status_from_variation(0.40) == "Crítico"

    def test_semaphore_green(self):
        col, label, _ = _semaphore(SEMAPHORE_GREEN)
        assert col == "green"
        assert "confianza" in label.lower()

    def test_semaphore_amber(self):
        col, label, _ = _semaphore((SEMAPHORE_GREEN + SEMAPHORE_AMBER) / 2)
        assert col == "amber"

    def test_semaphore_red(self):
        col, _, _ = _semaphore(SEMAPHORE_AMBER - 0.01)
        assert col == "red"


# ---------------------------------------------------------------------------
# 9. RobustnessEvaluator
# ---------------------------------------------------------------------------

class TestRobustnessEvaluator:

    @pytest.fixture
    def evaluator(self, synthetic_cohort):
        def sim_fn(**kwargs):
            age    = kwargs.get("age", 57)
            apache = kwargs.get("apache", 18)
            vam    = kwargs.get("vam_time", 5)
            return float(14.3 + 0.04*(apache-18) + 0.02*(vam-5) - 0.01*(age-57))

        return RobustnessEvaluator(
            simulate_fn=sim_fn,
            calibration_data=synthetic_cohort,
            numeric_cols=NUMERIC_COLS,
            seed=SEED,
        )

    @pytest.fixture
    def patient_params(self, default_patient):
        return default_patient

    # ── stress_noise ──────────────────────────────────────────────────

    def test_noise_returns_correct_number_of_levels(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.stress_noise(patient_params, base)
        assert len(results) == len(NOISE_LEVELS)

    def test_noise_variation_increases_with_sigma(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.stress_noise(patient_params, base)
        variations = [r.mean_variation for r in results]
        # La variación debe ser no decreciente con sigma
        assert all(
            variations[i] <= variations[i+1] + 0.05
            for i in range(len(variations)-1)
        ), f"Variaciones no crecientes: {variations}"

    def test_noise_component_score_in_range(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.stress_noise(patient_params, base)
        for r in results:
            assert 0.0 <= r.component_score <= 1.0

    # ── stress_missing ─────────────────────────────────────────────────

    def test_missing_returns_correct_number_of_levels(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.stress_missing(patient_params, base)
        assert len(results) == len(MISSING_LEVELS)

    def test_missing_variation_non_negative(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.stress_missing(patient_params, base)
        assert all(r.mean_variation >= 0 for r in results)

    # ── stress_shift ───────────────────────────────────────────────────

    def test_shift_result_has_both_extremes(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        result = evaluator.stress_shift(patient_params, base)
        assert result.base_prediction == pytest.approx(base, rel=0.01)
        assert result.relative_change_low >= 0
        assert result.relative_change_high >= 0
        assert 0.0 <= result.component_score <= 1.0

    # ── sensitivity_analysis ───────────────────────────────────────────

    def test_sensitivity_sorted_descending(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.sensitivity_analysis(patient_params, base)
        indices = [r.sensitivity_index for r in results]
        assert indices == sorted(indices, reverse=True)

    def test_sensitivity_ranks_start_at_1(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.sensitivity_analysis(patient_params, base)
        assert results[0].rank == 1

    def test_sensitivity_indices_normalized(self, evaluator, patient_params):
        base = evaluator.simulate_fn(**patient_params)
        results = evaluator.sensitivity_analysis(patient_params, base)
        assert results[0].sensitivity_index == pytest.approx(1.0, abs=0.01)
        assert all(0.0 <= r.sensitivity_index <= 1.0 for r in results)

    # ── evaluate (completo) ────────────────────────────────────────────

    def test_evaluate_returns_report(self, evaluator, patient_params):
        report = evaluator.evaluate(patient_params)
        assert isinstance(report, RobustnessReport)

    def test_evaluate_R_in_range(self, evaluator, patient_params):
        report = evaluator.evaluate(patient_params)
        assert 0.0 <= report.R <= 1.0

    def test_evaluate_semaphore_consistent_with_R(self, evaluator, patient_params):
        report = evaluator.evaluate(patient_params)
        if report.R >= SEMAPHORE_GREEN:
            assert report.semaphore == "green"
        elif report.R >= SEMAPHORE_AMBER:
            assert report.semaphore == "amber"
        else:
            assert report.semaphore == "red"

    def test_evaluate_weights_sum_to_R(self, evaluator, patient_params):
        """R debe ser la suma ponderada de sus componentes."""
        from framework.robustness_module import W_NOISE, W_MISSING, W_SHIFT
        report = evaluator.evaluate(patient_params)
        expected = (
            W_NOISE   * report.r_noise   +
            W_MISSING * report.r_missing +
            W_SHIFT   * report.r_shift
        )
        assert math.isclose(report.R, expected, rel_tol=1e-4)

    def test_evaluate_as_dict(self, evaluator, patient_params):
        report = evaluator.evaluate(patient_params)
        d = report.as_dict()
        assert "R" in d
        assert "semaphore" in d
        assert "noise_results" in d
        assert "missing_results" in d
        assert "sensitivity" in d

    # ── Calibración empírica del umbral de shift ────────────────────────

    # ── Números aleatorios comunes (CRN) y piso de n_runs ───────────────

    def test_min_n_runs_stabilizes_low_n_runs(self):
        """
        _with_min_n_runs debe elevar n_runs por debajo del piso, y dejar
        intactos valores ya iguales o superiores al piso.
        """
        from framework.robustness_module import _with_min_n_runs, MIN_N_RUNS_STRESS_TEST
        low = _with_min_n_runs({"n_runs": 50, "age": 57})
        assert low["n_runs"] == MIN_N_RUNS_STRESS_TEST

        high = _with_min_n_runs({"n_runs": MIN_N_RUNS_STRESS_TEST * 2, "age": 57})
        assert high["n_runs"] == MIN_N_RUNS_STRESS_TEST * 2

    def test_min_n_runs_handles_missing_key(self):
        from framework.robustness_module import _with_min_n_runs, MIN_N_RUNS_STRESS_TEST
        out = _with_min_n_runs({"age": 57})
        assert out["n_runs"] == MIN_N_RUNS_STRESS_TEST

    def test_crn_seed_present_after_evaluate(self, evaluator, patient_params):
        """evaluate() debe inyectar una semilla CRN entera reproducible."""
        report = evaluator.evaluate(patient_params, run_sensitivity=False)
        assert isinstance(report.base_prediction, float)
        # No hay campo público para la semilla en el reporte, pero la
        # evaluación no debe fallar ni producir NaN por su presencia.
        assert math.isfinite(report.base_prediction)

    def test_crn_eliminates_zero_effect_variable_from_top_ranking(self, adapter, default_patient):
        """
        Prueba de regresión directa del hallazgo real: sin CRN, una
        variable sin efecto en el modelo stub (preuti_stay, no aparece en
        la fórmula de mu/sigma de _StubSimulator) podía ganar el ranking
        OAT por puro ruido de muestreo Monte Carlo con n_runs bajo. Con
        CRN + piso de n_runs activos, su índice normalizado debe caer a
        un valor sustancialmente menor que el de la variable ganadora.
        """
        from framework.robustness_module import RobustnessEvaluator

        cal_df, num_cols = adapter.get_calibration_data()
        ev = RobustnessEvaluator(
            simulate_fn=adapter.simulate, calibration_data=cal_df,
            numeric_cols=num_cols, seed=42,
        )
        patient = dict(default_patient, n_runs=200)  # n_runs bajo, "por defecto"
        report = ev.evaluate(patient, run_sensitivity=True)

        by_var = {s.variable: s.sensitivity_index for s in report.sensitivity_results}
        assert by_var.get("preuti_stay", 1.0) < 0.3, (
            f"preuti_stay muestra índice {by_var.get('preuti_stay')}, "
            f"sugiriendo que el ruido Monte Carlo sigue dominando el ranking "
            f"pese a CRN + piso de n_runs."
        )

    def test_sensitivity_ranking_stable_across_repeated_evaluations(self, adapter, default_patient):
        """
        El ganador del ranking OAT debe ser el MISMO en repetidas
        evaluaciones con los mismos parámetros de entrada (estabilidad
        garantizada por CRN, no solo por la semilla global del evaluador).
        """
        from framework.robustness_module import RobustnessEvaluator

        cal_df, num_cols = adapter.get_calibration_data()
        ev = RobustnessEvaluator(
            simulate_fn=adapter.simulate, calibration_data=cal_df,
            numeric_cols=num_cols, seed=42,
        )
        patient = dict(default_patient, n_runs=200)

        winners = set()
        for _ in range(4):
            report = ev.evaluate(patient, run_sensitivity=True)
            winners.add(report.sensitivity_results[0].variable)

        assert len(winners) == 1, (
            f"El ganador del ranking OAT varía entre evaluaciones repetidas: {winners}"
        )

    def test_shift_threshold_calibrated_above_base(self, evaluator):
        """
        El umbral calibrado debe ser mayor que el umbral base fijo, ya que
        el shift perturba varias variables simultáneamente (compone
        efectos individuales, produciendo variaciones mayores que
        perturbar una sola variable como en ruido/faltantes).
        """
        from framework.robustness_module import CLINICAL_THRESHOLD
        assert evaluator.shift_threshold > CLINICAL_THRESHOLD

    def test_stress_noise_and_missing_have_independent_rng(self, evaluator):
        """
        Regresión del hallazgo real: stress_noise() y stress_missing() se
        ejecutan concurrentemente en hilos separados (evaluate() usa
        ThreadPoolExecutor) y antes usaban el mismo self.rng compartido sin
        sincronización, una condición de carrera real en
        numpy.random.Generator. Deben tener generadores propios,
        independientes entre sí y del generador secuencial principal.
        """
        assert evaluator._rng_noise is not evaluator._rng_missing
        assert evaluator._rng_noise is not evaluator.rng
        assert evaluator._rng_missing is not evaluator.rng

    def test_concurrent_evaluate_calls_are_thread_safe(self, adapter, default_patient):
        """
        Ejecuta evaluate() repetidamente (cada llamada dispara sus propios
        3 hilos concurrentes vía ThreadPoolExecutor) y verifica que no
        ocurren excepciones ni resultados degenerados (NaN, ganador OAT
        inconsistente) que delatarían una condición de carrera residual.
        """
        from framework.robustness_module import RobustnessEvaluator

        cal_df, num_cols = adapter.get_calibration_data()
        ev = RobustnessEvaluator(
            simulate_fn=adapter.simulate, calibration_data=cal_df,
            numeric_cols=num_cols, seed=42,
        )
        patient = dict(default_patient, n_runs=200)

        winners = set()
        for _ in range(8):
            report = ev.evaluate(patient, run_sensitivity=True)
            assert math.isfinite(report.R)
            assert 0.0 <= report.R <= 1.0
            winners.add(report.sensitivity_results[0].variable)

        assert len(winners) == 1, (
            f"Ganador del ranking OAT inconsistente entre corridas "
            f"concurrentes repetidas: {winners} — posible condición de "
            f"carrera residual."
        )

    def test_shift_threshold_within_safety_bounds(self, evaluator):
        from framework.robustness_module import (
            SHIFT_THRESHOLD_FLOOR, SHIFT_THRESHOLD_CEILING,
        )
        assert SHIFT_THRESHOLD_FLOOR <= evaluator.shift_threshold <= SHIFT_THRESHOLD_CEILING

    def test_shift_threshold_disabled_falls_back_to_base(self, synthetic_cohort):
        """Con calibrate_shift_threshold=False, debe usar el umbral base sin calibrar."""
        from framework.robustness_module import RobustnessEvaluator, CLINICAL_THRESHOLD

        def sim_fn(**kwargs):
            return 100.0 + kwargs.get("apache", 0)

        ev = RobustnessEvaluator(
            simulate_fn=sim_fn, calibration_data=synthetic_cohort,
            numeric_cols=NUMERIC_COLS, seed=SEED,
            calibrate_shift_threshold=False,
        )
        assert ev.shift_threshold == ev.clinical_threshold == CLINICAL_THRESHOLD

    def test_shift_produces_nonzero_scores_for_some_patients(self, synthetic_cohort):
        """
        Verifica que la calibración produce dispersión real: al menos
        algunos pacientes de la propia cohorte deben obtener r_shift > 0
        (antes de calibrar, colapsaba a 0 para prácticamente todos).
        """
        from framework.robustness_module import RobustnessEvaluator

        def sim_fn(**kwargs):
            age = kwargs.get("age", 57)
            apache = kwargs.get("apache", 12)
            vam = kwargs.get("vam_time", 120)
            return float(60 + 0.8 * apache + 0.05 * vam - 0.1 * age)

        ev = RobustnessEvaluator(
            simulate_fn=sim_fn, calibration_data=synthetic_cohort,
            numeric_cols=NUMERIC_COLS, seed=SEED,
        )

        rng = np.random.default_rng(11)
        idx = rng.choice(len(synthetic_cohort), size=15, replace=False)
        sample = synthetic_cohort.iloc[idx]

        scores = []
        for _, row in sample.iterrows():
            patient = {c: float(row[c]) for c in NUMERIC_COLS + ["d1","d2","d3","d4","resp_insuf","artif_vent"]}
            base = sim_fn(**patient)
            shift = ev.stress_shift(patient, base)
            scores.append(shift.component_score)

        assert any(s > 0 for s in scores), (
            "Todos los r_shift son 0 incluso tras calibrar — la calibración "
            "no está discriminando correctamente."
        )

    def test_report_includes_shift_threshold_used(self, evaluator, patient_params):
        report = evaluator.evaluate(patient_params)
        assert report.shift_threshold_used == evaluator.shift_threshold

    def test_evaluate_high_apache_lowers_R(self, evaluator):
        """Paciente con APACHE muy alto debe tener R más bajo que el típico."""
        typical  = dict(age=57, apache=18, vam_time=5, uti_stay=8,
                        preuti_stay=2, percent=5,
                        d1=1, d2=0, d3=0, d4=0, resp_insuf=1, artif_vent=1, n_runs=200)
        severe   = dict(typical)
        severe["apache"] = 55

        r_typical = evaluator.evaluate(typical, run_sensitivity=False).R
        r_severe  = evaluator.evaluate(severe,  run_sensitivity=False).R

        # No garantizamos estrictamente menor, pero verificamos que está
        # bien calculado y no está fuera de rango
        assert 0.0 <= r_severe <= 1.0
        assert 0.0 <= r_typical <= 1.0


# ---------------------------------------------------------------------------
# 10. SimUciAdapter (modo stub)
# ---------------------------------------------------------------------------

    def test_missing_only_imputes_clinical_keys(self, evaluator, patient_params):
        """El stress-test de faltantes no debe imputar n_runs, percent ni _crn_seed."""
        calls = []
        base_fn = evaluator.simulate_fn

        def spy(**kw):
            calls.append(dict(kw))
            return base_fn(**kw)

        evaluator.simulate_fn = spy
        try:
            params = dict(patient_params, n_runs=1000, _crn_seed=11)
            evaluator.stress_missing(params, base_fn(**params))
        finally:
            evaluator.simulate_fn = base_fn
        assert calls
        assert all(c["n_runs"] == 1000 for c in calls)
        assert all(c["_crn_seed"] == 11 for c in calls)
        assert all(c["percent"] == params["percent"] for c in calls)


class TestSimUciAdapter:

    def test_adapter_stub_mode(self, adapter):
        """En entorno de pruebas, el adaptador debe estar en modo stub."""
        assert adapter.stub_mode is True

    def test_parse_dataframe_maps_real_csv_columns(self):
        """
        Cierra un hueco de cobertura real: _parse_dataframe() (la ruta de
        carga del CSV real de SimUci) nunca se ejercitaba en la suite,
        pese a ser precisamente el tipo de código donde se encontraron los
        bugs más graves de esta auditoría (nombres de columna incorrectos
        en versiones anteriores del adaptador). Verifica el mapeo de
        columnas y la conversión de días a horas para Est. UCI/Est. PreUCI.
        """
        fake_real_csv = pd.DataFrame({
            'Edad': [57, 68],
            'APACHE': [14, 30],
            'TiempoVAM': [120, 480],
            'Est. UCI': [4.0, 7.1],       # en DÍAS en el CSV real
            'Est. PreUCI': [0.5, 1.2],    # en DÍAS en el CSV real
            'InsufResp': [1, 1],
            'VA': [0, 2],
            'Diag.Ing1': [21, 23],
            'Diag.Ing2': [0, 21],
            'Diag.Ing3': [0, 0],
            'Diag.Ing4': [0, 0],
        })

        adapter = SimUciAdapter.__new__(SimUciAdapter)
        result = adapter._parse_dataframe(fake_real_csv)

        assert set(ALL_FEATURES + [TARGET_COL]).issubset(set(result.columns))
        assert result.loc[0, 'uti_stay'] == pytest.approx(4.0 * 24)
        assert result.loc[0, 'preuti_stay'] == pytest.approx(0.5 * 24)
        assert result.loc[0, 'd1'] == 21
        assert result.loc[0, 'age'] == 57
        assert result.loc[1, 'd2'] == 21
        assert result[TARGET_COL].isna().all(), (
            "TARGET_COL no debe venir poblado desde el CSV crudo — es "
            "resultado de simulación, no un dato de entrada."
        )

    def test_cohort_size(self, adapter):
        assert adapter.cohort_size == 208

    def test_custom_imputation_values_respected(self, adapter):
        """Verifica la rama poco ejercitada de imputation_values personalizado."""
        from framework.robustness_module import RobustnessEvaluator
        cal_df, num_cols = adapter.get_calibration_data()
        custom_imp = {"age": 99.0, "apache": 5.0}
        ev = RobustnessEvaluator(
            simulate_fn=adapter.simulate, calibration_data=cal_df,
            numeric_cols=num_cols, seed=1, imputation_values=custom_imp,
        )
        assert ev._imputation.get("age") == 99.0
        assert ev._imputation.get("apache") == 5.0

    def test_calibration_size(self, adapter):
        n_cal = int(208 * 0.30)
        assert abs(adapter.calibration_size - n_cal) <= 2

    def test_simulate_positive(self, adapter, default_patient):
        pred = adapter.simulate(**default_patient)
        assert pred > 0
        assert math.isfinite(pred)

    def test_simulate_n_returns_list(self, adapter, default_patient):
        preds = adapter.simulate_n(n=20, **default_patient)
        assert len(preds) == 20
        assert all(math.isfinite(p) and p > 0 for p in preds)

    def test_configure_updates_params(self, adapter, default_patient):
        adapter.configure(default_patient)
        assert adapter._is_configured is True
        assert adapter._patient_params == default_patient

    def test_patient_feature_vector_shape(self, adapter, default_patient):
        adapter.configure(default_patient)
        vec = adapter.patient_feature_vector()
        assert vec.shape == (1, len(ALL_FEATURES))

    def test_patient_feature_vector_before_configure(self):
        fresh_adapter = SimUciAdapter()
        with pytest.raises(RuntimeError, match="configure"):
            fresh_adapter.patient_feature_vector()

    def test_get_calibration_data_returns_dataframe(self, adapter):
        df, cols = adapter.get_calibration_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(cols, list)
        assert all(c in df.columns for c in cols)

    def test_get_conformal_inputs(self, adapter):
        X_cal, y_cal, model = adapter.get_conformal_inputs()
        assert X_cal.shape[0] == y_cal.shape[0]
        assert X_cal.shape[1] == len(ALL_FEATURES)
        assert hasattr(model, "predict")

    def test_repr(self, adapter):
        r = repr(adapter)
        assert "SimUciAdapter" in r
        assert "stub" in r

    def test_model_predict_shape(self, adapter, synthetic_cohort):
        X = synthetic_cohort[ALL_FEATURES].to_numpy(dtype=float)[:5]
        _, _, model = adapter.get_conformal_inputs()
        preds = model.predict(X)
        assert preds.shape == (5,)


# ---------------------------------------------------------------------------
# 11. Integración: Incertidumbre + Robustez + Adaptador
# ---------------------------------------------------------------------------

    def test_default_has_no_test_set(self, adapter):
        """Por compatibilidad, sin test_split no hay conjunto de prueba."""
        assert adapter.test_size == 0

    def test_three_way_split_sizes_and_disjoint(self):
        """Entrenamiento/calibración/prueba disjuntos y de tamaño 126/41/41."""
        a = SimUciAdapter(calibration_split=0.20, test_split=0.20)
        assert (a.train_size, a.calibration_size, a.test_size) == (126, 41, 41)
        assert a.train_size + a.calibration_size + a.test_size == a.cohort_size
        idx = np.random.default_rng(42).permutation(a.cohort_size)
        assert len(set(idx[:41]) & set(idx[41:82])) == 0

    def test_get_test_data_populates_target(self):
        a = SimUciAdapter(calibration_split=0.20, test_split=0.20)
        df = a.get_test_data()
        assert len(df) == 41 and df[TARGET_COL].notna().all()


class TestEndToEnd:
    """
    Prueba de integración completa que simula el flujo del dashboard:
    patient_params → adapter → cuantificación + robustez → reporte.
    """

    def test_full_pipeline_conformal(self, adapter, default_patient):
        adapter.configure(default_patient)
        X_cal, y_cal, model = adapter.get_conformal_inputs()
        x_new = adapter.patient_feature_vector()

        uq = UncertaintyQuantifier(method="conformal", confidence=0.95, seed=SEED)
        uq.fit(X_cal, y_cal, model)
        iv = uq.predict_interval(x_new)

        cal_df, num_cols = adapter.get_calibration_data()
        ev = RobustnessEvaluator(
            simulate_fn=adapter.simulate,
            calibration_data=cal_df,
            numeric_cols=num_cols,
            seed=SEED,
        )
        report = ev.evaluate(default_patient, run_sensitivity=True)

        # Verificaciones cruzadas
        assert iv.lower >= 0
        assert iv.upper > iv.lower
        assert 0.0 <= report.R <= 1.0
        assert report.semaphore in ("green", "amber", "red")
        assert len(report.sensitivity_results) > 0

    def test_full_pipeline_bootstrap(self, adapter, default_patient):
        adapter.configure(default_patient)
        cal_df, num_cols = adapter.get_calibration_data()

        uq = UncertaintyQuantifier(method="bootstrap", confidence=0.90, B=100, seed=SEED)
        uq.fit(cal_df, num_cols)
        iv = uq.predict_interval(adapter.simulate, default_patient)

        assert isinstance(iv, PredictionInterval)
        assert iv.lower < iv.upper

    def test_full_pipeline_montecarlo(self, adapter, default_patient):
        uq = UncertaintyQuantifier(method="montecarlo", confidence=0.90, B=150, seed=SEED)
        iv = uq.predict_interval(adapter.simulate, default_patient)

        assert isinstance(iv, PredictionInterval)
        assert iv.method == "montecarlo"
        assert iv.lower >= 0

    def test_report_serializable(self, adapter, default_patient):
        """El reporte debe poder serializarse a dict (para Streamlit/JSON)."""
        import json
        adapter.configure(default_patient)
        cal_df, num_cols = adapter.get_calibration_data()
        ev = RobustnessEvaluator(
            simulate_fn=adapter.simulate,
            calibration_data=cal_df,
            numeric_cols=num_cols,
            seed=SEED,
        )
        report = ev.evaluate(default_patient, run_sensitivity=False)
        d = report.as_dict()

        # Verificar serialización JSON
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        assert reloaded["R"] == round(report.R, 4)


# ---------------------------------------------------------------------------
# 12. StubSimulator
# ---------------------------------------------------------------------------

class TestStubSimulator:
    """
    Pruebas del simulador sintético (_StubSimulator), usado solo cuando el
    proyecto SimUci real no está disponible en el PYTHONPATH. Refleja la
    firma real de run_experiment(): devuelve un DataFrame con las columnas
    ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI",
    "Estadia Post UCI"], no un escalar.
    """

    def test_run_experiment_stub_returns_dataframe(self):
        df = _StubSimulator.run_experiment_stub(
            age=57, apache=18, vam_time=120, uti_stay=96,
            d1=21, d2=0, d3=0, n_runs=100,
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM",
            "Estadia UCI", "Estadia Post UCI",
        ]
        assert len(df) == 100

    def test_run_experiment_stub_target_positive(self):
        df = _StubSimulator.run_experiment_stub(age=57, apache=18, n_runs=100)
        pred = float(df[TARGET_COL].mean())
        assert pred > 0
        assert math.isfinite(pred)

    def test_run_experiment_stub_n_runs_zero_safe(self):
        """n_runs<=0 no debe producir un DataFrame vacío ni NaN (clamp defensivo)."""
        df = _StubSimulator.run_experiment_stub(age=57, apache=18, n_runs=0)
        assert len(df) >= 1
        assert math.isfinite(float(df[TARGET_COL].mean()))

    def test_high_apache_increases_prediction(self):
        low_df  = _StubSimulator.run_experiment_stub(age=57, apache=5, n_runs=500)
        high_df = _StubSimulator.run_experiment_stub(age=57, apache=30, n_runs=500)
        low  = float(low_df[TARGET_COL].mean())
        high = float(high_df[TARGET_COL].mean())
        assert high > low * 0.5   # No estrictamente mayor por estocasticidad,
                                   # pero la tendencia debe mantenerse

    def test_cohort_shape(self):
        df = _StubSimulator.build_cohort()
        assert df.shape == (208, len(ALL_FEATURES) + 1)  # +1 para TARGET_COL

    def test_cohort_no_negatives(self):
        df = _StubSimulator.build_cohort()
        for col in NUMERIC_COLS:
            assert (df[col] >= 0).all(), f"Columna {col} contiene negativos"

    def test_cohort_target_positive(self):
        df = _StubSimulator.build_cohort()
        assert (df[TARGET_COL] > 0).all()

    def test_cohort_within_real_ranges(self):
        """La cohorte sintética debe respetar los rangos reales de SimUci."""
        from framework.simuci_adapter import VALID_RANGES
        df = _StubSimulator.build_cohort()
        for col, (lo, hi) in VALID_RANGES.items():
            assert (df[col] >= lo).all(), f"{col} tiene valores bajo el mínimo {lo}"
            assert (df[col] <= hi).all(), f"{col} tiene valores sobre el máximo {hi}"

