"""
robustness_module.py
====================
Módulo de evaluación de robustez para el modelo SimUci.

Implementa tres tipos de stress-tests:
  1. Ruido gaussiano aditivo  N(0, sigma * s_j)  sobre variables numéricas
  2. Sustitución de variables por valores de imputación (datos faltantes)
  3. Desplazamiento poblacional (dataset shift) a percentiles p10 / p90

El resultado central es el Índice de Robustez R ∈ [0, 1] con semáforo
clínico (verde / ámbar / rojo) y pesos determinados empíricamente.

Compatibilidad con SimUci
-------------------------
La función de predicción recibe los mismos kwargs que ``run_experiment()``
del proyecto SimUci: n_runs, age, d1..d4, apache, resp_insuf, artif_vent,
vam_time, uti_stay, preuti_stay, percent.

Autor   : Jorge Jesús Santos García
Tutora  : DrC. Deborah Raquel Galpert Cañizares
Centro  : UCLV — Departamento de Ciencias de la Computación
Año     : 2025
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constantes del framework
# ---------------------------------------------------------------------------

# Niveles de ruido gaussiano evaluados
NOISE_LEVELS: tuple[float, ...] = (0.10, 0.25, 0.50)

# Porcentajes de datos faltantes evaluados
MISSING_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

# Percentiles para el stress-test de shift poblacional
SHIFT_PERCENTILES: tuple[float, float] = (10.0, 90.0)

# Umbrales del semáforo clínico
SEMAPHORE_GREEN: float = 0.70   # R >= 0.70 → Alta confianza
SEMAPHORE_AMBER: float = 0.40   # R >= 0.40 → Confianza moderada
                                 # R <  0.40 → Baja confianza

# Pesos del índice R: valores PROPUESTOS por criterio de diseño (ruido y datos
# faltantes con igual peso por ser perturbaciones de un solo dato; shift con
# menor peso por ser una perturbación conjunta más severa). NO fueron estimados
# a partir de datos ni de desenlaces clínicos; no existe un procedimiento de
# optimización asociado. Su efecto se evalúa en el análisis de sensibilidad
# a los pesos de experimentos/ejecutar_experimentos.py.
W_NOISE: float = 0.40
W_MISSING: float = 0.40
W_SHIFT: float = 0.20

# Umbral clínico: variación relativa considerada crítica
CLINICAL_THRESHOLD: float = 0.25   # 25 % de variación relativa

# Calibración empírica del umbral para el stress-test de shift poblacional.
#
# A diferencia de los stress-tests de ruido y datos faltantes (que
# perturban variables de forma individual o parcial), el stress-test de
# shift desplaza TODAS las variables numéricas SIMULTÁNEAMENTE a sus
# percentiles p10/p90 — una perturbación conjunta que, por composición de
# efectos individuales, produce sistemáticamente variaciones relativas
# mucho mayores que CLINICAL_THRESHOLD (0.25). Usar el mismo umbral fijo
# para ambos casos hace que r_shift colapse casi siempre a 0 para
# cualquier paciente, sin aportar señal discriminativa al índice R
# (Expresión 22).
#
# Se calibra en su lugar un umbral específico para este test, estimado
# empíricamente como un percentil de la distribución de variaciones
# relativas observadas al aplicar el propio stress-test de shift sobre
# una muestra de la cohorte de calibración — reutilizando la misma
# filosofía de calibración basada en datos que sustenta la conformal
# prediction (Vovk et al., 2005) en vez de un umbral arbitrario.
SHIFT_THRESHOLD_PERCENTILE: float = 50.0   # percentil usado para calibrar
SHIFT_CALIBRATION_SAMPLE_SIZE: int = 30    # pacientes muestreados para calibrar
                                            # (probado empíricamente: n=30 da la
                                            # mejor estabilidad std≈0.07 con costo
                                            # despreciable, ~0.03s en modo stub)
SHIFT_CALIBRATION_N_RUNS: int = 200        # réplicas para calibrar (probado
                                            # empíricamente: n=200 da la mejor
                                            # estabilidad std≈0.02 con costo
                                            # despreciable, ~0.03s en modo stub;
                                            # n=30 daba std≈0.05, más ruidoso)
SHIFT_THRESHOLD_FLOOR: float = 0.25        # nunca calibrar por debajo del umbral base
SHIFT_THRESHOLD_CEILING: float = 1.50      # límite superior de seguridad

# Réplicas por nivel en el stress-test de ruido
N_REPS_NOISE: int = 100

# Piso mínimo de n_runs usado INTERNAMENTE por los stress-tests y el
# análisis de sensibilidad OAT, independiente del n_runs que el usuario
# configure para la predicción principal (vía UncertaintyQuantifier).
#
# Motivación (hallazgo empírico): comparar dos simulaciones de Monte
# Carlo con n_runs pequeño (p. ej. 100-300, valores típicos en la barra
# lateral del dashboard) introduce error estándar de la media que puede
# ser MAYOR que el efecto real de perturbar una variable. Esto hace que
# variables sin efecto real en el modelo (ruido puro) puedan aparentar
# ser "las más sensibles" solo por azar en el muestreo, contaminando el
# ranking OAT (Expresión 16-17) y los stress-tests de ruido/faltantes.
#
# Verificado empíricamente: con n_runs=100-300 el ranking OAT cambia de
# forma inestable entre ejecuciones; a partir de n_runs≈2000-10000 las
# variables con efecto real (según los coeficientes del modelo) dominan
# consistentemente el ranking. Se usa 1000 como piso práctico de
# compromiso entre estabilidad estadística y costo computacional.
#
# Nota: la solución estadísticamente más rigurosa sería usar números
# aleatorios comunes (common random numbers, CRN — Law & Kelton, 2000)
# para aparear las simulaciones base y perturbada con el mismo flujo de
# aleatoriedad, cancelando el ruido en la diferencia. Esto requeriría que
# simulate_fn acepte una semilla explícita por llamada, lo cual no está
# garantizado para el motor real de SimUci; el piso de n_runs es la
# mitigación practicable sin esa garantía.
MIN_N_RUNS_STRESS_TEST: int = 1000


def _with_min_n_runs(params: dict, min_n_runs: int = MIN_N_RUNS_STRESS_TEST) -> dict:
    """Devuelve una copia de params con n_runs elevado al piso mínimo si es necesario."""
    out = dict(params)
    current = int(out.get("n_runs", 0) or 0)
    out["n_runs"] = max(current, min_n_runs)
    return out

# Tipo del semáforo
SemaphoreColor = Literal["green", "amber", "red"]


# ---------------------------------------------------------------------------
# Contenedores de resultados
# ---------------------------------------------------------------------------

@dataclass
class NoiseStressResult:
    """Resultado del stress-test de ruido gaussiano."""
    level: float                        # σ evaluado
    mean_variation: float               # variación relativa media
    status: str                         # "Estable" / "Alerta" / "Crítico"
    component_score: float              # r_ruido para este nivel


@dataclass
class MissingStressResult:
    """Resultado del stress-test de datos faltantes."""
    missing_pct: float                  # porcentaje de variables ausentes
    mean_variation: float               # variación relativa media
    status: str
    component_score: float              # r_faltante para este nivel


@dataclass
class ShiftStressResult:
    """Resultado del stress-test de shift poblacional."""
    percentile_low: float               # predicción con variables en p10
    percentile_high: float              # predicción con variables en p90
    base_prediction: float
    relative_change_low: float
    relative_change_high: float
    component_score: float              # r_shift


@dataclass
class SensitivityResult:
    """Análisis de sensibilidad OAT por variable."""
    variable: str
    sensitivity_index: float            # índice normalizado ∈ [0, 1]
    rank: int                           # posición (1 = más influyente)


@dataclass
class RobustnessReport:
    """Informe completo de robustez para un caso clínico."""

    # Índice global
    R: float
    semaphore: SemaphoreColor
    semaphore_label: str
    semaphore_description: str

    # Componentes del índice R
    r_noise: float
    r_missing: float
    r_shift: float

    # Resultados detallados por stress-test
    noise_results: list[NoiseStressResult] = field(default_factory=list)
    missing_results: list[MissingStressResult] = field(default_factory=list)
    shift_result: ShiftStressResult | None = None
    sensitivity_results: list[SensitivityResult] = field(default_factory=list)

    # Metadatos
    base_prediction: float = 0.0
    clinical_threshold: float = CLINICAL_THRESHOLD
    shift_threshold_used: float = CLINICAL_THRESHOLD

    def as_dict(self) -> dict:
        return {
            "R": round(self.R, 4),
            "semaphore": self.semaphore,
            "semaphore_label": self.semaphore_label,
            "r_noise": round(self.r_noise, 4),
            "r_missing": round(self.r_missing, 4),
            "r_shift": round(self.r_shift, 4),
            "base_prediction": round(self.base_prediction, 4),
            "noise_results": [
                {
                    "sigma": r.level,
                    "variation": round(r.mean_variation, 4),
                    "status": r.status,
                }
                for r in self.noise_results
            ],
            "missing_results": [
                {
                    "pct": r.missing_pct,
                    "variation": round(r.mean_variation, 4),
                    "status": r.status,
                }
                for r in self.missing_results
            ],
            "sensitivity": [
                {"variable": s.variable, "index": round(s.sensitivity_index, 4)}
                for s in self.sensitivity_results
            ],
        }


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _relative_variation(base: float, perturbed: float) -> float:
    """Variación relativa respecto a la predicción base."""
    denom = abs(base) if abs(base) > 1e-9 else 1.0
    return abs(perturbed - base) / denom


def _status_from_variation(variation: float) -> str:
    if variation < 0.10:
        return "Estable"
    elif variation < CLINICAL_THRESHOLD:
        return "Alerta"
    return "Crítico"


def _component_score(mean_variation: float, threshold: float = CLINICAL_THRESHOLD) -> float:
    """Convierte variación relativa en puntuación de robustez ∈ [0, 1]."""
    return max(0.0, 1.0 - min(1.0, mean_variation / threshold))


def _semaphore(R: float) -> tuple[SemaphoreColor, str, str]:
    """Devuelve color, etiqueta y descripción del semáforo."""
    if R >= SEMAPHORE_GREEN:
        return (
            "green",
            "Alta confianza",
            "El modelo SimUci es estable ante las perturbaciones evaluadas. "
            "La predicción puede usarse como referencia clínica con confianza.",
        )
    elif R >= SEMAPHORE_AMBER:
        return (
            "amber",
            "Confianza moderada",
            "El modelo muestra sensibilidad moderada. "
            "Use el intervalo de predicción junto al juicio clínico directo.",
        )
    return (
        "red",
        "Baja confianza",
        "Alta fragilidad detectada. Este caso puede estar fuera del dominio "
        "de validez del modelo. Priorice el criterio médico experto.",
    )


# ---------------------------------------------------------------------------
# Clase principal: RobustnessEvaluator
# ---------------------------------------------------------------------------

class RobustnessEvaluator:
    """
    Evaluador de robustez para el modelo SimUci.

    Ejecuta los tres stress-tests en paralelo (concurrent.futures) y calcula
    el índice de robustez R ponderado.

    Parameters
    ----------
    simulate_fn : Callable
        Función que acepta los parámetros del paciente y devuelve una
        predicción escalar (float).  Debe ser compatible con la firma de
        ``run_experiment()`` de SimUci o con un wrapper que la llame.
    calibration_data : pd.DataFrame
        Cohorte de referencia usada para estimar las desviaciones estándar
        de cada variable y los percentiles p10/p90 (su tamaño depende de la
        partición configurada en el adaptador).
    numeric_cols : list[str]
        Columnas numéricas del DataFrame de calibración.
    imputation_values : dict[str, Any] | None
        Valores de imputación por columna (mediana para numéricas,
        moda para categóricas).  Si None se calculan automáticamente.
    n_jobs : int
        Número de hilos para la ejecución paralela (default: 3).
    seed : int | None
        Semilla para reproducibilidad.
    clinical_threshold : float
        Umbral de variación relativa considerado clínicamente relevante.

    Examples
    --------
    >>> ev = RobustnessEvaluator(simulate_fn=my_sim, calibration_data=df_cal,
    ...                          numeric_cols=["age","apache","vam_time"])
    >>> report = ev.evaluate(patient_params)
    >>> print(report.R, report.semaphore)
    0.75  'green'
    """

    def __init__(
        self,
        simulate_fn: Callable[..., float],
        calibration_data: pd.DataFrame,
        numeric_cols: list[str],
        imputation_values: dict[str, Any] | None = None,
        n_jobs: int = 3,
        seed: int | None = None,
        clinical_threshold: float = CLINICAL_THRESHOLD,
        calibrate_shift_threshold: bool = True,
        target_col: str | None = None,
    ):
        self.simulate_fn = simulate_fn
        self.calibration_data = calibration_data.copy()
        self.numeric_cols = [c for c in numeric_cols if c in calibration_data.columns]
        self.n_jobs = n_jobs

        # Generadores aleatorios independientes por contexto de ejecución.
        #
        # stress_noise() y stress_missing() se ejecutan CONCURRENTEMENTE en
        # hilos separados (ThreadPoolExecutor, ver evaluate()). Un único
        # numpy.random.Generator compartido entre hilos sin sincronización
        # externa no es seguro para llamadas concurrentes: el estado interno
        # del generador puede leerse/actualizarse de forma inconsistente,
        # comprometiendo silenciosamente tanto la validez estadística de los
        # números generados como la reproducibilidad garantizada por la
        # semilla (afectando, en particular, la estabilidad del ranking OAT
        # lograda mediante CRN). Se usa numpy.random.SeedSequence.spawn()
        # (patrón recomendado por la documentación de NumPy para paralelismo)
        # para derivar generadores hijos independientes y sin solapamiento
        # de flujo, uno por cada método que se ejecuta en su propio hilo.
        seed_seq = np.random.SeedSequence(seed)
        rng_main, rng_noise, rng_missing = [
            np.random.default_rng(s) for s in seed_seq.spawn(3)
        ]
        self.rng = rng_main          # uso secuencial: calibración, CRN, etc.
        self._rng_noise = rng_noise    # uso exclusivo de stress_noise()
        self._rng_missing = rng_missing  # uso exclusivo de stress_missing()

        self.clinical_threshold = clinical_threshold

        # Columna objetivo a excluir al reconstruir parámetros de paciente
        # desde una fila de calibration_data (usada solo por la calibración
        # del umbral de shift). Si no se indica explícitamente, se asume
        # por convención que es la última columna del DataFrame — así el
        # módulo permanece agnóstico del adaptador concreto (SimUciAdapter
        # u otro) que haya construido calibration_data.
        self.target_col = target_col or self.calibration_data.columns[-1]

        # Estadísticas de calibración
        self._std: dict[str, float] = {}
        self._p10: dict[str, float] = {}
        self._p90: dict[str, float] = {}
        self._compute_calibration_stats()

        # Valores de imputación
        if imputation_values:
            self._imputation = imputation_values
        else:
            self._imputation = self._compute_imputation_values()

        # Umbral calibrado empíricamente para el stress-test de shift
        # (ver SHIFT_THRESHOLD_PERCENTILE). Si la calibración falla o se
        # desactiva, se usa clinical_threshold como respaldo.
        self.shift_threshold: float = self.clinical_threshold
        if calibrate_shift_threshold:
            try:
                self.shift_threshold = self._calibrate_shift_threshold()
            except Exception as exc:
                warnings.warn(
                    f"Calibración del umbral de shift falló ({exc}). "
                    f"Usando umbral base {self.clinical_threshold}."
                )

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _compute_calibration_stats(self):
        """Calcula std, p10 y p90 para cada variable numérica."""
        for col in self.numeric_cols:
            vals = self.calibration_data[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                self._std[col] = 1.0
                self._p10[col] = 0.0
                self._p90[col] = 1.0
                continue
            self._std[col] = float(np.std(vals, ddof=1)) or 1.0
            self._p10[col] = float(np.percentile(vals, SHIFT_PERCENTILES[0]))
            self._p90[col] = float(np.percentile(vals, SHIFT_PERCENTILES[1]))

    def _compute_imputation_values(self) -> dict[str, Any]:
        """Mediana para numéricas; 0 para el resto."""
        imp: dict[str, Any] = {}
        for col in self.calibration_data.columns:
            if col in self.numeric_cols:
                imp[col] = float(self.calibration_data[col].median())
            else:
                try:
                    imp[col] = int(self.calibration_data[col].mode().iloc[0])
                except Exception:
                    imp[col] = 0
        return imp

    def _calibrate_shift_threshold(self) -> float:
        """
        Calibra empíricamente el umbral de variación relativa usado por
        ``stress_shift()``.

        Motivación
        ----------
        El stress-test de shift desplaza TODAS las variables numéricas a
        la vez (p10/p90), a diferencia de los tests de ruido y datos
        faltantes. Esta perturbación conjunta compone efectos individuales
        y produce sistemáticamente variaciones relativas mucho mayores que
        el umbral base (0.25), colapsando r_shift a 0 para prácticamente
        cualquier paciente y anulando su aporte discriminativo al índice R.

        Procedimiento
        -------------
        1. Se muestrean hasta SHIFT_CALIBRATION_SAMPLE_SIZE pacientes de la
           cohorte de calibración.
        2. Para cada uno, se ejecuta el propio stress-test de shift (con
           n_runs reducido a SHIFT_CALIBRATION_N_RUNS para mantener el
           costo computacional acotado) y se registra su variación
           relativa máxima (p10 vs p90).
        3. El umbral calibrado es el percentil SHIFT_THRESHOLD_PERCENTILE
           de esa distribución empírica, recortado a
           [SHIFT_THRESHOLD_FLOOR, SHIFT_THRESHOLD_CEILING].

        Con el percentil 50 (mediana) por defecto, aproximadamente la
        mitad de los pacientes de la cohorte obtienen r_shift > 0 y la
        otra mitad r_shift < 1, dando al componente una distribución
        informativa en vez de un valor constante.

        Returns
        -------
        float — umbral calibrado (en las mismas unidades que
        ``_relative_variation``, es decir, fracción no porcentaje).
        """
        n = min(SHIFT_CALIBRATION_SAMPLE_SIZE, len(self.calibration_data))
        if n < 3:
            return self.clinical_threshold

        idx = self.rng.choice(len(self.calibration_data), size=n, replace=False)
        sample = self.calibration_data.iloc[idx]

        variations: list[float] = []
        for _, row in sample.iterrows():
            params = {col: row[col] for col in self.calibration_data.columns
                      if col != self.target_col and col in row.index}
            params = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                      for k, v in params.items()}
            params["n_runs"] = SHIFT_CALIBRATION_N_RUNS
            # CRN: misma semilla para base y p10/p90 de este paciente, ver
            # nota completa en evaluate(). Cada paciente muestreado usa una
            # semilla propia (independiente entre pacientes).
            params["_crn_seed"] = int(self.rng.integers(0, 2**31))

            try:
                base = float(self.simulate_fn(**params))
                shift_result = self.stress_shift(params, base, _threshold_override=1e9)
                # threshold_override=huge para que component_score no trunque
                # la variación real (solo queremos medirla, no puntuarla aquí)
                max_var = max(shift_result.relative_change_low,
                              shift_result.relative_change_high)
                if np.isfinite(max_var):
                    variations.append(max_var)
            except Exception:
                continue

        if len(variations) < 3:
            return self.clinical_threshold

        calibrated = float(np.percentile(variations, SHIFT_THRESHOLD_PERCENTILE))
        return float(np.clip(calibrated, SHIFT_THRESHOLD_FLOOR, SHIFT_THRESHOLD_CEILING))

    # ------------------------------------------------------------------
    # Stress-test 1: Ruido gaussiano
    # ------------------------------------------------------------------

    def stress_noise(
        self,
        patient_params: dict,
        base_pred: float,
        levels: tuple[float, ...] = NOISE_LEVELS,
    ) -> list[NoiseStressResult]:
        """
        Stress-test de ruido gaussiano N(0, sigma * s_j).

        Parameters
        ----------
        patient_params : dict
            Parámetros de entrada del paciente.
        base_pred : float
            Predicción sin perturbación (referencia).
        levels : tuple[float]
            Niveles de sigma a evaluar (default: 0.10, 0.25, 0.50).

        Returns
        -------
        list[NoiseStressResult]
        """
        results: list[NoiseStressResult] = []

        for sigma in levels:
            variations: list[float] = []

            for _ in range(N_REPS_NOISE):
                perturbed = dict(patient_params)
                for col in self.numeric_cols:
                    if col not in perturbed:
                        continue
                    s_j = self._std.get(col, 1.0)
                    noise = float(self._rng_noise.normal(0, sigma * s_j))
                    perturbed[col] = max(0, float(perturbed[col]) + noise)

                try:
                    pred = float(self.simulate_fn(**perturbed))
                    variations.append(_relative_variation(base_pred, pred))
                except Exception:
                    variations.append(0.0)

            mean_var = float(np.mean(variations)) if variations else 0.0
            results.append(
                NoiseStressResult(
                    level=sigma,
                    mean_variation=mean_var,
                    status=_status_from_variation(mean_var),
                    component_score=_component_score(mean_var),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Stress-test 2: Datos faltantes
    # ------------------------------------------------------------------

    def stress_missing(
        self,
        patient_params: dict,
        base_pred: float,
        levels: tuple[float, ...] = MISSING_LEVELS,
    ) -> list[MissingStressResult]:
        """
        Stress-test de datos faltantes.

        Sustituye aleatoriamente un porcentaje de las variables por sus
        valores de imputación y mide la variación de la predicción.

        Parameters
        ----------
        patient_params : dict
            Parámetros de entrada del paciente.
        base_pred : float
            Predicción de referencia sin perturbación.
        levels : tuple[float]
            Proporciones de variables ausentes a evaluar.

        Returns
        -------
        list[MissingStressResult]
        """
        # Solo las variables CLÍNICAS pueden "faltar". Los parámetros de
        # configuración de la simulación (n_runs, percent) y la semilla de
        # números aleatorios comunes (_crn_seed) no son datos del paciente:
        # imputarlos (p. ej. n_runs -> 0) invalidaría la simulación y rompería
        # el apareamiento CRN, inflando artificialmente la variación.
        all_keys = [
            k for k in patient_params
            if k in self._imputation and k != self.target_col
        ]
        if not all_keys:
            all_keys = list(patient_params.keys())
        results: list[MissingStressResult] = []

        for pct in levels:
            n_missing = max(1, int(round(len(all_keys) * pct)))
            variations: list[float] = []

            for _ in range(50):   # 50 muestras por nivel
                perturbed = dict(patient_params)
                missing_keys = self._rng_missing.choice(
                    all_keys, size=n_missing, replace=False
                ).tolist()
                for k in missing_keys:
                    perturbed[k] = self._imputation.get(k, 0)

                try:
                    pred = float(self.simulate_fn(**perturbed))
                    variations.append(_relative_variation(base_pred, pred))
                except Exception:
                    variations.append(0.0)

            mean_var = float(np.mean(variations)) if variations else 0.0
            results.append(
                MissingStressResult(
                    missing_pct=pct,
                    mean_variation=mean_var,
                    status=_status_from_variation(mean_var),
                    component_score=_component_score(mean_var),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Stress-test 3: Shift poblacional
    # ------------------------------------------------------------------

    def stress_shift(
        self,
        patient_params: dict,
        base_pred: float,
        _threshold_override: float | None = None,
    ) -> ShiftStressResult:
        """
        Stress-test de shift poblacional.

        Desplaza cada variable numérica al percentil p10 y p90 de la
        cohorte de calibración y mide la variación de la predicción.

        El componente de robustez usa ``self.shift_threshold`` (calibrado
        empíricamente en el constructor, ver ``_calibrate_shift_threshold``)
        en vez del umbral clínico base, ya que esta perturbación conjunta
        de todas las variables produce variaciones sistemáticamente
        mayores que las de ruido o datos faltantes.

        Parameters
        ----------
        _threshold_override : float | None
            Uso interno: permite a ``_calibrate_shift_threshold()`` medir
            la variación real sin que ``component_score`` la trunque
            durante la propia fase de calibración. No usar directamente.

        Returns
        -------
        ShiftStressResult
        """
        # Extremo inferior (p10)
        params_low = dict(patient_params)
        for col in self.numeric_cols:
            if col in params_low:
                params_low[col] = max(0, self._p10.get(col, 0))
        try:
            pred_low = float(self.simulate_fn(**params_low))
        except Exception:
            pred_low = base_pred

        # Extremo superior (p90)
        params_high = dict(patient_params)
        for col in self.numeric_cols:
            if col in params_high:
                params_high[col] = max(0, self._p90.get(col, 0))
        try:
            pred_high = float(self.simulate_fn(**params_high))
        except Exception:
            pred_high = base_pred

        rel_low = _relative_variation(base_pred, pred_low)
        rel_high = _relative_variation(base_pred, pred_high)
        max_var = max(rel_low, rel_high)

        threshold = _threshold_override if _threshold_override is not None \
            else getattr(self, "shift_threshold", self.clinical_threshold)

        return ShiftStressResult(
            percentile_low=pred_low,
            percentile_high=pred_high,
            base_prediction=base_pred,
            relative_change_low=rel_low,
            relative_change_high=rel_high,
            component_score=_component_score(max_var, threshold=threshold),
        )

    # ------------------------------------------------------------------
    # Análisis de sensibilidad OAT
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        patient_params: dict,
        base_pred: float,
        delta_fraction: float = 0.10,
    ) -> list[SensitivityResult]:
        """
        Análisis de sensibilidad One-At-a-Time (OAT).

        Varía cada variable numérica en ±delta_fraction de su desviación
        estándar calibrada y mide el cambio en la predicción.

        Parameters
        ----------
        patient_params : dict
            Parámetros base del paciente.
        base_pred : float
            Predicción base sin perturbación.
        delta_fraction : float
            Fracción de la std usada como perturbación (default: 0.10).

        Returns
        -------
        list[SensitivityResult]  ordenados por índice descendente.
        """
        raw_indices: dict[str, float] = {}

        for col in self.numeric_cols:
            if col not in patient_params:
                continue
            s_j = self._std.get(col, 1.0)
            delta = delta_fraction * s_j

            variations: list[float] = []
            for sign in (+1, -1):
                perturbed = dict(patient_params)
                perturbed[col] = max(0, float(patient_params[col]) + sign * delta)
                try:
                    pred = float(self.simulate_fn(**perturbed))
                    variations.append(_relative_variation(base_pred, pred))
                except Exception:
                    variations.append(0.0)

            raw_indices[col] = float(np.mean(variations))

        # Normalizar al máximo
        max_idx = max(raw_indices.values(), default=1.0) or 1.0
        sorted_items = sorted(raw_indices.items(), key=lambda x: x[1], reverse=True)

        return [
            SensitivityResult(
                variable=col,
                sensitivity_index=round(val / max_idx, 4),
                rank=rank + 1,
            )
            for rank, (col, val) in enumerate(sorted_items)
        ]

    # ------------------------------------------------------------------
    # Evaluación completa
    # ------------------------------------------------------------------

    def evaluate(
        self,
        patient_params: dict,
        run_sensitivity: bool = True,
    ) -> RobustnessReport:
        """
        Ejecuta la batería completa de stress-tests y calcula el índice R.

        Los tres stress-tests se ejecutan en paralelo con ThreadPoolExecutor
        para mantener el tiempo de respuesta dentro de los 10 segundos
        especificados en los requisitos no funcionales.

        Parameters
        ----------
        patient_params : dict
            Parámetros clínicos del paciente con las mismas claves que
            acepta ``run_experiment()`` de SimUci.
        run_sensitivity : bool
            Si True, también ejecuta el análisis OAT (default: True).

        Returns
        -------
        RobustnessReport  con índice R, semáforo y resultados detallados.
        """
        # Estabilizar n_runs para todos los stress-tests internos (ver
        # MIN_N_RUNS_STRESS_TEST): evita que el ruido de muestreo Monte
        # Carlo domine sobre el efecto real de las perturbaciones cuando
        # el usuario configuró un n_runs pequeño para la predicción
        # principal. No afecta la predicción mostrada al usuario, que se
        # calcula por separado en UncertaintyQuantifier.
        stable_params = _with_min_n_runs(patient_params)

        # Números aleatorios comunes (CRN, Law y Kelton, 2000): se fija una
        # única semilla para la predicción base y todas las perturbaciones
        # de esta evaluación, de modo que comparten la misma aleatoriedad
        # interna de la simulación. Así, la diferencia observada entre
        # base y perturbado refleja principalmente el efecto real de la
        # perturbación, no una nueva realización independiente del ruido
        # Monte Carlo. Solo tiene efecto en modo stub: el run_experiment()
        # real de SimUci no expone un parámetro de semilla, por lo que en
        # modo real esta clave es ignorada de forma segura (ver
        # SimUciAdapter.simulate(), que extrae parámetros por nombre en
        # la rama real y nunca reenvía claves desconocidas).
        stable_params["_crn_seed"] = int(self.rng.integers(0, 2**31))

        # Predicción base
        try:
            base_pred = float(self.simulate_fn(**stable_params))
        except Exception as exc:
            warnings.warn(f"simulate_fn falló para el caso base: {exc}")
            base_pred = 0.0

        # Ejecución paralela de los tres stress-tests
        noise_results: list[NoiseStressResult] = []
        missing_results: list[MissingStressResult] = []
        shift_result: ShiftStressResult | None = None

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            fut_noise = executor.submit(
                self.stress_noise, stable_params, base_pred
            )
            fut_missing = executor.submit(
                self.stress_missing, stable_params, base_pred
            )
            fut_shift = executor.submit(
                self.stress_shift, stable_params, base_pred
            )

            noise_results = fut_noise.result()
            missing_results = fut_missing.result()
            shift_result = fut_shift.result()

        # Análisis de sensibilidad (opcional, secuencial)
        sensitivity_results: list[SensitivityResult] = []
        if run_sensitivity:
            try:
                sensitivity_results = self.sensitivity_analysis(
                    stable_params, base_pred
                )
            except Exception as exc:
                warnings.warn(f"Análisis de sensibilidad falló: {exc}")

        # Componentes del índice R
        # Usa el nivel sigma=0.25 (índice 1) como representativo del ruido
        r_noise = noise_results[1].component_score if len(noise_results) > 1 else 0.5
        # Usa el nivel 10% de faltantes como representativo
        r_missing = missing_results[1].component_score if len(missing_results) > 1 else 0.5
        r_shift = shift_result.component_score if shift_result else 0.5

        # Índice R ponderado
        R = round(W_NOISE * r_noise + W_MISSING * r_missing + W_SHIFT * r_shift, 4)
        R = min(1.0, max(0.0, R))

        color, label, description = _semaphore(R)

        return RobustnessReport(
            R=R,
            semaphore=color,
            semaphore_label=label,
            semaphore_description=description,
            r_noise=r_noise,
            r_missing=r_missing,
            r_shift=r_shift,
            noise_results=noise_results,
            missing_results=missing_results,
            shift_result=shift_result,
            sensitivity_results=sensitivity_results,
            base_prediction=base_pred,
            clinical_threshold=self.clinical_threshold,
            shift_threshold_used=getattr(self, "shift_threshold", self.clinical_threshold),
        )
