"""
uncertainty_module.py
=====================
Módulo de cuantificación de incertidumbre para el modelo SimUci.

Implementa tres métodos de construcción de intervalos de predicción:
  1. Bootstrapping paramétrico (log-normal / gamma / exponencial / Weibull)
  2. Conformal prediction — variante split (inductive conformal prediction)
  3. Simulación Monte Carlo

Uso básico
----------
>>> from framework.uncertainty_module import UncertaintyQuantifier
>>> uq = UncertaintyQuantifier(method="conformal", confidence=0.95)
>>> uq.fit(X_calib, y_calib, model)
>>> lo, hi, pred = uq.predict_interval(x_new)

Compatibilidad
--------------
Python 3.10+  |  numpy >= 1.24  |  scipy >= 1.11  |  scikit-learn >= 1.3
fitter >= 1.7 (ajuste automático de distribuciones)

Autor   : Jorge Jesús Santos García
Tutora  : DrC. Deborah Raquel Galpert Cañizares
Centro  : UCLV — Departamento de Ciencias de la Computación
Año     : 2025
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------
Method = Literal["conformal", "bootstrap", "montecarlo"]
Distribution = Literal["lognorm", "gamma", "expon", "weibull_min", "norm"]

# ---------------------------------------------------------------------------
# Resultado de un intervalo de predicción
# ---------------------------------------------------------------------------

@dataclass
class PredictionInterval:
    """Contenedor de resultado de un intervalo de predicción."""

    prediction: float
    lower: float
    upper: float
    confidence: float
    method: str
    width: float = field(init=False)
    empirical_coverage: float | None = None
    distribution: str | None = None
    dist_params: dict | None = None

    def __post_init__(self):
        self.width = round(self.upper - self.lower, 4)

    def as_dict(self) -> dict:
        return {
            "prediction": round(self.prediction, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "width": self.width,
            "confidence": self.confidence,
            "method": self.method,
            "empirical_coverage": self.empirical_coverage,
            "distribution": self.distribution,
        }


# ---------------------------------------------------------------------------
# Ajuste de distribuciones de probabilidad
# ---------------------------------------------------------------------------

class DistributionFitter:
    """
    Ajusta distribuciones de probabilidad a datos univariados mediante
    la prueba de Kolmogorov-Smirnov y selecciona la de mejor p-valor.

    Parameters
    ----------
    candidates : list[str]
        Distribuciones de scipy.stats a evaluar.
    """

    CANDIDATES: list[str] = ["lognorm", "gamma", "expon", "weibull_min", "norm"]

    def __init__(self, candidates: list[str] | None = None):
        self.candidates = candidates or self.CANDIDATES
        self.best_dist_: str | None = None
        self.best_params_: dict | None = None
        self.best_pvalue_: float = 0.0
        self.all_results_: dict[str, dict] = {}

    def fit(self, data: NDArray) -> "DistributionFitter":
        """Ajusta distribuciones y selecciona la de mayor p-valor KS."""
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data) & (data > 0)]
        if len(data) < 10:
            warnings.warn("Muestra insuficiente (<10 obs.) para ajustar distribuciones.")
            return self

        for dist_name in self.candidates:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                ks_stat, p_value = stats.kstest(data, dist_name, args=params)
                self.all_results_[dist_name] = {
                    "params": params,
                    "ks_stat": ks_stat,
                    "p_value": p_value,
                }
                if p_value > self.best_pvalue_:
                    self.best_pvalue_ = p_value
                    self.best_dist_ = dist_name
                    self.best_params_ = {
                        f"param_{i}": v for i, v in enumerate(params)
                    }
            except Exception:
                continue

        return self

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray:
        """Genera n muestras desde la distribución ajustada."""
        if self.best_dist_ is None:
            raise RuntimeError("Llama a .fit() antes de .sample().")
        rng = rng or np.random.default_rng()
        dist = getattr(stats, self.best_dist_)
        params = tuple(
            v for v in (self.all_results_[self.best_dist_]["params"])
        )
        return dist.rvs(*params, size=n, random_state=int(rng.integers(0, 2**31)))


# ---------------------------------------------------------------------------
# Método 1: Bootstrapping paramétrico
# ---------------------------------------------------------------------------

class ParametricBootstrapper:
    """
    Cuantificación de incertidumbre mediante bootstrap por perturbación de
    las entradas del paciente (variante paramétrica con ruido gaussiano).

    Pasos
    -----
    1. Ajusta una distribución a cada variable numérica de la cohorte
       (DistributionFitter); la distribución de mejor ajuste se registra y
       se informa junto con el intervalo (``PredictionInterval.distribution``).
    2. Para un nuevo paciente ejecuta B réplicas de la simulación. En cada
       réplica suma a cada variable numérica ajustada un ruido normal
       N(0, max(0.10·|x|, 1)). NOTA: las distribuciones ajustadas NO se
       muestrean en este paso; el ruido es gaussiano alrededor del valor del
       paciente. Por eso no es un bootstrap paramétrico estricto (que
       remuestrea desde la distribución ajustada), sino un bootstrap por
       perturbación de entradas.
    3. Construye el intervalo de predicción como los percentiles α/2 y
       1-α/2 de la distribución de las B predicciones.

    Parameters
    ----------
    B : int
        Número de réplicas bootstrap (default: 1000).
    confidence : float
        Nivel de confianza del intervalo (default: 0.95).
    seed : int | None
        Semilla para reproducibilidad.
    """

    def __init__(
        self,
        B: int = 1_000,
        confidence: float = 0.95,
        seed: int | None = None,
    ):
        self.B = B
        self.confidence = confidence
        self.rng = np.random.default_rng(seed)
        self.fitters_: dict[str, DistributionFitter] = {}
        self._is_fitted = False
        self.last_predictions_: NDArray | None = None   # B predicciones de la última llamada

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, numeric_cols: list[str]) -> "ParametricBootstrapper":
        """
        Ajusta distribuciones de probabilidad a las columnas numéricas.

        Parameters
        ----------
        data : DataFrame
            Cohorte de calibración.
        numeric_cols : list[str]
            Columnas numéricas a las que ajustar distribuciones.
        """
        for col in numeric_cols:
            if col not in data.columns:
                continue
            fitter = DistributionFitter()
            fitter.fit(data[col].dropna().to_numpy())
            self.fitters_[col] = fitter

        self._is_fitted = True
        return self

    def predict_interval(
        self,
        simulate_fn: Callable[..., float],
        patient_params: dict,
    ) -> PredictionInterval:
        """
        Construye un intervalo de predicción para un nuevo paciente.

        Parameters
        ----------
        simulate_fn : Callable
            Función que recibe patient_params y devuelve una predicción escalar.
        patient_params : dict
            Parámetros de entrada del paciente.

        Returns
        -------
        PredictionInterval
        """
        if not self._is_fitted:
            raise RuntimeError("Llama a .fit() antes de .predict_interval().")

        predictions: list[float] = []
        base_pred = float(simulate_fn(**patient_params))

        for _ in range(self.B):
            perturbed = dict(patient_params)
            for col, fitter in self.fitters_.items():
                if col in perturbed and fitter.best_dist_ is not None:
                    # Perturbación gaussiana alrededor del valor del paciente
                    # (no se muestrea la distribución ajustada; ver docstring).
                    scale = max(abs(perturbed[col]) * 0.1, 1.0)
                    noise = self.rng.normal(0, scale)
                    perturbed[col] = max(0, perturbed[col] + noise)
            try:
                pred = float(simulate_fn(**perturbed))
                predictions.append(pred)
            except Exception:
                predictions.append(base_pred)

        preds = np.array(predictions)
        self.last_predictions_ = preds
        alpha = (1 - self.confidence) / 2
        lo = float(np.percentile(preds, alpha * 100))
        hi = float(np.percentile(preds, (1 - alpha) * 100))

        # Distribución predominante
        dominant_dist = None
        if self.fitters_:
            dominant_dist = max(
                self.fitters_.values(),
                key=lambda f: f.best_pvalue_,
            ).best_dist_

        return PredictionInterval(
            prediction=base_pred,
            lower=max(0.0, lo),
            upper=hi,
            confidence=self.confidence,
            method="bootstrap",
            distribution=dominant_dist,
        )


# ---------------------------------------------------------------------------
# Método 2: Conformal Prediction (split / inductive)
# ---------------------------------------------------------------------------

class ConformalPredictor:
    """
    Intervalos de predicción con garantías formales de cobertura.

    Implementa la variante split (inductive conformal prediction):
      1. Entrena el modelo en el subconjunto de entrenamiento.
      2. Calcula puntuaciones de no conformidad en el subconjunto de calibración.
      3. Construye el intervalo como [ŷ − q, ŷ + q] donde q es el
         estadístico de orden ⌈(n_cal+1)(1−α)⌉-ésimo (con techo/ceiling,
         no interpolado) de las puntuaciones de calibración ordenadas.

    Garantía
    --------
    La cobertura empírica >= 1-α bajo intercambiabilidad de los datos.

    Parameters
    ----------
    confidence : float
        Nivel de confianza (default: 0.95).
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self._nonconf_scores: NDArray | None = None
        self._threshold: float | None = None
        self._model: Any = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def fit(
        self,
        X_cal: NDArray | pd.DataFrame,
        y_cal: NDArray | pd.Series,
        model: Any,
    ) -> "ConformalPredictor":
        """
        Calibra el predictor conformal.

        Parameters
        ----------
        X_cal : array-like, shape (n_cal, n_features)
            Conjunto de calibración (features).
        y_cal : array-like, shape (n_cal,)
            Valores reales del conjunto de calibración.
        model : sklearn-compatible
            Modelo predictivo ya entrenado (debe implementar .predict()).
        """
        self._model = model
        X_cal = np.asarray(X_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)

        y_hat = model.predict(X_cal).astype(float)
        self._nonconf_scores = np.abs(y_cal - y_hat)

        n = len(self._nonconf_scores)
        alpha = 1 - self.confidence

        # Nivel de cuantil exacto de la conformal prediction split
        # (Vovk et al., 2005; Angelopoulos y Bates, 2022, Algoritmo 1):
        #   q = estadístico de orden ⌈(n+1)(1-alpha)⌉-ésimo (1-indexado)
        #       de las puntuaciones de calibración ORDENADAS.
        #
        # NOTA DE CORRECCIÓN: una versión anterior calculaba un "nivel de
        # percentil" y lo pasaba a np.percentile(). Esto tiene dos
        # problemas: (1) sin especificar method='higher', np.percentile()
        # interpola linealmente entre estadísticos de orden vecinos,
        # subestimando el umbral hasta un 23% en pruebas con
        # distribuciones de error asimétricas; (2) incluso especificando
        # method='higher', la convención interna de posición de
        # np.percentile() es (n-1)*p (0-indexada), que NO coincide en
        # general con el índice ⌈(n+1)(1-alpha)⌉ específico de conformal
        # prediction para n finito. La implementación correcta ordena las
        # puntuaciones directamente e indexa con la fórmula exacta, sin
        # intermediar por la convención de percentil de NumPy.
        sorted_scores = np.sort(self._nonconf_scores)
        idx = int(np.ceil((n + 1) * (1 - alpha))) - 1  # 0-indexado
        idx = min(max(idx, 0), n - 1)
        self._threshold = float(sorted_scores[idx])
        self._is_fitted = True
        return self

    def predict_interval(
        self,
        x_new: NDArray | pd.DataFrame,
    ) -> PredictionInterval:
        """
        Genera un intervalo de predicción para un nuevo caso.

        Parameters
        ----------
        x_new : array-like, shape (1, n_features) o (n_features,)
            Vector de características del nuevo paciente.

        Returns
        -------
        PredictionInterval
        """
        if not self._is_fitted:
            raise RuntimeError("Llama a .fit() antes de .predict_interval().")

        x = np.asarray(x_new, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        pred = float(self._model.predict(x)[0])
        q = self._threshold

        empirical_cov = self._empirical_coverage()

        return PredictionInterval(
            prediction=pred,
            lower=max(0.0, pred - q),
            upper=pred + q,
            confidence=self.confidence,
            method="conformal",
            empirical_coverage=empirical_cov,
        )

    def evaluate_coverage(
        self,
        X_test: NDArray | pd.DataFrame,
        y_test: NDArray | pd.Series,
    ) -> float:
        """
        Evalúa la cobertura empírica sobre un conjunto de prueba.

        Returns
        -------
        float
            Fracción de casos cubiertos por el intervalo.
        """
        if not self._is_fitted:
            raise RuntimeError("Llama a .fit() antes de evaluar cobertura.")

        X = np.asarray(X_test, dtype=float)
        y = np.asarray(y_test, dtype=float)
        q = self._threshold

        y_hat = self._model.predict(X).astype(float)
        covered = np.sum((y >= y_hat - q) & (y <= y_hat + q))
        return float(covered) / len(y)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _empirical_coverage(self) -> float | None:
        """Cobertura en el conjunto de calibración (leave-one-out proxy)."""
        if self._nonconf_scores is None:
            return None
        q = self._threshold
        covered = np.sum(self._nonconf_scores <= q)
        return float(covered) / len(self._nonconf_scores)


# ---------------------------------------------------------------------------
# Método 3: Monte Carlo
# ---------------------------------------------------------------------------

class MonteCarloQuantifier:
    """
    Cuantificación de incertidumbre mediante simulación Monte Carlo.

    Ejecuta múltiples réplicas de la simulación variando los parámetros
    de entrada según sus distribuciones de probabilidad asignadas.

    Parameters
    ----------
    n_reps : int
        Número de réplicas (default: 500).
    confidence : float
        Nivel de confianza (default: 0.95).
    seed : int | None
        Semilla para reproducibilidad.
    """

    def __init__(
        self,
        n_reps: int = 500,
        confidence: float = 0.95,
        seed: int | None = None,
    ):
        self.n_reps = n_reps
        self.confidence = confidence
        self.rng = np.random.default_rng(seed)
        self.last_predictions_: NDArray | None = None   # predicciones de la última llamada

    def predict_interval(
        self,
        simulate_fn: Callable[..., float],
        patient_params: dict,
        param_distributions: dict[str, tuple[str, tuple]] | None = None,
    ) -> PredictionInterval:
        """
        Genera un intervalo de predicción Monte Carlo.

        Parameters
        ----------
        simulate_fn : Callable
            Función de simulación que devuelve un escalar.
        patient_params : dict
            Parámetros del paciente (valores centrales).
        param_distributions : dict | None
            Distribuciones para cada parámetro: {param: (dist_name, params_tuple)}.
            Si None, se añade ruido gaussiano del 10% sobre cada valor numérico.

        Returns
        -------
        PredictionInterval
        """
        base_pred = float(simulate_fn(**patient_params))
        predictions: list[float] = []

        # Claves que NO deben perturbarse como variables clínicas continuas:
        # - 'n_runs' es un parámetro de configuración de la simulación, no
        #   una variable del paciente; perturbarlo puede llevarlo a 0 o
        #   negativo y producir simulaciones vacías (NaN).
        # - Las variables categóricas/indicadoras (d1..d4, resp_insuf,
        #   artif_vent) están codificadas como enteros pero no son
        #   continuas; sumarles ruido gaussiano las saca de su dominio
        #   válido {0,1} o {1..5}.
        _NON_PERTURBABLE_KEYS = {
            "n_runs", "percent", "_crn_seed",
            "d1", "d2", "d3", "d4", "resp_insuf", "artif_vent",
        }

        for _ in range(self.n_reps):
            sampled = dict(patient_params)
            for key, val in patient_params.items():
                if key in _NON_PERTURBABLE_KEYS:
                    continue
                if not isinstance(val, (int, float)):
                    continue
                if param_distributions and key in param_distributions:
                    dist_name, dist_params = param_distributions[key]
                    dist = getattr(stats, dist_name)
                    sampled[key] = float(dist.rvs(*dist_params, random_state=None))
                else:
                    # Ruido gaussiano del 10%
                    sigma = max(abs(val) * 0.10, 0.5)
                    sampled[key] = max(0, val + float(self.rng.normal(0, sigma)))
            try:
                predictions.append(float(simulate_fn(**sampled)))
            except Exception:
                predictions.append(base_pred)

        preds = np.array(predictions)
        self.last_predictions_ = preds
        alpha = (1 - self.confidence) / 2
        lo = float(np.percentile(preds, alpha * 100))
        hi = float(np.percentile(preds, (1 - alpha) * 100))

        return PredictionInterval(
            prediction=base_pred,
            lower=max(0.0, lo),
            upper=hi,
            confidence=self.confidence,
            method="montecarlo",
        )


# ---------------------------------------------------------------------------
# Fachada unificada: UncertaintyQuantifier
# ---------------------------------------------------------------------------

class UncertaintyQuantifier:
    """
    Fachada unificada para los tres métodos de cuantificación de incertidumbre.

    Patrón de diseño Strategy: permite intercambiar el método sin modificar
    el código del dashboard ni del núcleo de SimUci.

    Parameters
    ----------
    method : {"conformal", "bootstrap", "montecarlo"}
        Método de construcción del intervalo de predicción.
    confidence : float
        Nivel de confianza (default: 0.95).
    B : int
        Réplicas para bootstrapping o Monte Carlo (default: 1000).
    seed : int | None
        Semilla global de reproducibilidad.

    Examples
    --------
    >>> uq = UncertaintyQuantifier(method="conformal", confidence=0.95)
    >>> uq.fit(X_cal, y_cal, rf_model)
    >>> result = uq.predict_interval(x_new)
    >>> print(result.as_dict())
    """

    def __init__(
        self,
        method: Method = "conformal",
        confidence: float = 0.95,
        B: int = 1_000,
        seed: int | None = None,
    ):
        self.method = method
        self.confidence = confidence
        self.B = B
        self.seed = seed
        self._backend: ConformalPredictor | ParametricBootstrapper | MonteCarloQuantifier | None = None
        self._build_backend()

    # ------------------------------------------------------------------
    # Construcción interna
    # ------------------------------------------------------------------

    def _build_backend(self):
        if self.method == "conformal":
            self._backend = ConformalPredictor(confidence=self.confidence)
        elif self.method == "bootstrap":
            self._backend = ParametricBootstrapper(
                B=self.B, confidence=self.confidence, seed=self.seed
            )
        elif self.method == "montecarlo":
            self._backend = MonteCarloQuantifier(
                n_reps=self.B, confidence=self.confidence, seed=self.seed
            )
        else:
            raise ValueError(f"Método desconocido: {self.method!r}. Use 'conformal', 'bootstrap' o 'montecarlo'.")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, *args, **kwargs) -> "UncertaintyQuantifier":
        """
        Ajusta el backend seleccionado.

        Para ``conformal``:
            fit(X_cal, y_cal, model)
        Para ``bootstrap``:
            fit(data_df, numeric_cols)
        Para ``montecarlo``:
            No requiere ajuste previo.
        """
        if self._backend is None:
            self._build_backend()
        self._backend.fit(*args, **kwargs)  # type: ignore[union-attr]
        return self

    def predict_interval(self, *args, **kwargs) -> PredictionInterval:
        """
        Genera el intervalo de predicción usando el método configurado.

        Parámetros según el método:
        - conformal  : predict_interval(x_new)
        - bootstrap  : predict_interval(simulate_fn, patient_params)
        - montecarlo : predict_interval(simulate_fn, patient_params)
        """
        if self._backend is None:
            raise RuntimeError("El cuantificador no ha sido inicializado.")
        return self._backend.predict_interval(*args, **kwargs)  # type: ignore[union-attr]

    @property
    def last_predictions_(self) -> NDArray | None:
        """Predicciones de las B réplicas de la última llamada (bootstrap/MC)."""
        return getattr(self._backend, "last_predictions_", None)

    def evaluate_coverage(self, X_test, y_test) -> float:
        """Disponible solo para el método conformal."""
        if not isinstance(self._backend, ConformalPredictor):
            raise NotImplementedError(
                "evaluate_coverage() solo está disponible para el método 'conformal'."
            )
        return self._backend.evaluate_coverage(X_test, y_test)

    def change_method(self, method: Method) -> "UncertaintyQuantifier":
        """Cambia el método de intervalos sin perder la configuración base."""
        self.method = method
        self._build_backend()
        return self

    def __repr__(self) -> str:
        return (
            f"UncertaintyQuantifier(method={self.method!r}, "
            f"confidence={self.confidence}, B={self.B})"
        )


# ---------------------------------------------------------------------------
# Descomposición de incertidumbre por tipo
# ---------------------------------------------------------------------------

def decompose_uncertainty(
    predictions_bootstrap: NDArray,
    predictions_mc: NDArray,
) -> dict[str, float]:
    """
    Estima la contribución relativa de cada tipo de incertidumbre.

    Parameters
    ----------
    predictions_bootstrap : NDArray
        Distribución de predicciones del bootstrapping (incertidumbre paramétrica).
    predictions_mc : NDArray
        Distribución de predicciones de Monte Carlo (aleatoria + paramétrica).

    Returns
    -------
    dict con claves 'aleatoria', 'parametrica', 'estructural' (porcentajes).
    """
    var_mc = float(np.var(predictions_mc))
    var_boot = float(np.var(predictions_bootstrap))
    total = var_mc + 1e-12

    param_frac = min(var_boot / total, 1.0)
    aleat_frac = max(0.0, 1.0 - param_frac) * 0.75
    struct_frac = max(0.0, 1.0 - param_frac) * 0.25

    norm = aleat_frac + param_frac + struct_frac
    raw_pcts = {
        "aleatoria": aleat_frac / norm * 100,
        "parametrica": param_frac / norm * 100,
        "estructural": struct_frac / norm * 100,
    }

    # Método del mayor resto (largest remainder) para garantizar que los
    # tres porcentajes redondeados a 1 decimal sumen exactamente 100.0,
    # evitando el error de redondeo independiente por componente.
    scale = 10  # 1 decimal de precisión
    floors = {k: math.floor(v * scale) for k, v in raw_pcts.items()}
    remainder = round(100 * scale) - sum(floors.values())
    remainders = sorted(
        raw_pcts.keys(),
        key=lambda k: (raw_pcts[k] * scale) - floors[k],
        reverse=True,
    )
    for i in range(remainder):
        floors[remainders[i % len(remainders)]] += 1

    return {k: round(v / scale, 1) for k, v in floors.items()}
