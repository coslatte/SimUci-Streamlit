"""
simuci_adapter.py
=================
Adaptador entre el framework de incertidumbre/robustez y el núcleo real
del proyecto SimUci (SimUci-Streamlit).

Este archivo fue corregido tras inspeccionar directamente el código fuente
real del proyecto (utils/helpers/helpers.py, utils/constants/*.py). Las
versiones anteriores de este adaptador asumían una API genérica
`from simuci import run_experiment, ...` que NO existe en el proyecto real;
las funciones reales viven en ``utils.helpers`` y tienen firmas, rangos y
semántica distintas a las asumidas originalmente. Ver notas de corrección
al final de este docstring.

Firmas reales verificadas (utils/helpers/helpers.py)
------------------------------------------------------
    run_experiment(n_runs, age, d1, d2, d3, d4, apache, resp_insuf,
                    artif_vent, vam_time, uti_stay, preuti_stay,
                    percent=10, debug=False) -> pd.DataFrame

    Devuelve un DataFrame con columnas:
        ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM",
         "Estadia UCI", "Estadia Post UCI"]

    predict(df) -> tuple[np.ndarray, np.ndarray]
        Requiere columnas: {'Diag.Ing1','Diag.Ing2','Diag.Egr2',
                             'TiempoVAM','APACHE','Edad'}
        Carga el modelo desde utils.constants.paths.PREDICTION_MODEL_PATH
        (models/prediction_model.joblib), con fallback a Google Drive.

    get_data_for_prediction(data: dict | DataFrame) -> DataFrame

Rangos reales verificados (utils/constants/limits.py)
------------------------------------------------------
    AGE_MIN=14, AGE_MAX=100
    APACHE_MIN=0, APACHE_MAX=36
    VAM_T_MIN=24, VAM_T_MAX=700           (¡en HORAS, no en días!)
    UTI_STAY_MIN=0, UTI_STAY_MAX=200      (horas)
    PREUTI_STAY_MIN=0, PREUTI_STAY_MAX=34 (horas)
    SIM_RUNS_MIN=50, SIM_RUNS_MAX=100_000, SIM_RUNS_DEFAULT=200
    SIM_PERCENT_MIN=0, SIM_PERCENT_MAX=10, SIM_PERCENT_DEFAULT=3

Notas importantes sobre la semántica de los parámetros
--------------------------------------------------------
- ``d1, d2, d3, d4`` NO son indicadores binarios de diagnóstico. Son
  CÓDIGOS de diagnóstico (0-40) tomados del catálogo PREUCI_DIAG, uno por
  cada uno de los hasta 4 diagnósticos de ingreso del paciente. Por tanto
  no deben perturbarse como variables numéricas continuas en los
  stress-tests de robustez.
- ``resp_insuf`` es un código 0-5 del catálogo RESP_INSUF (no una variable
  numérica continua).
- ``artif_vent`` es un código 0-2 del catálogo VENTILATION_TYPE (no una
  variable numérica continua).
- ``percent`` es un parámetro de CONFIGURACIÓN de la simulación (0-10, por
  defecto 3) que gobierna la proporción interna con la que el motor de
  simulación divide el tiempo total de VAM entre las fases "Pre VAM" y
  "Post VAM". NO es una variable clínica de estadía del paciente y, al
  igual que ``n_runs``, no debe perturbarse como si lo fuera.
- La variable de salida de interés clínico usada por este framework es
  "Estadia Post UCI" (columna literal del DataFrame devuelto por
  run_experiment), que representa el tiempo de estadía posterior al
  egreso de UCI.

Corrección respecto a versiones anteriores de este archivo
-------------------------------------------------------------
1. El import ``from simuci import run_experiment, multiple_replication,
   load_model`` era incorrecto: ``run_experiment`` es una función definida
   en ``utils.helpers`` (no en el paquete ``simuci``), y no existe una
   función ``load_model`` en ninguno de los dos; la carga del modelo se
   hace inline con ``joblib.load()`` dentro de ``predict()``.
2. Las rutas ``_MODEL_PATH`` y ``_COHORT_PATH`` apuntaban a nombres de
   archivo que no coinciden con los reales
   (``models/random_forest.joblib`` vs. el real
   ``models/prediction_model.joblib``; y
   ``data/Ficherodedatos(MO)17-1-2023.csv`` vs. el real
   ``data/fichero_datos_MO_17_1_2023.csv``). Ambas quedan corregidas.
3. Los rangos de APACHE (0-71 asumido vs. 0-36 real) y de TiempoVAM
   (0-120 días asumido vs. 24-700 HORAS real) han sido corregidos.
4. ``d1..d4``, ``resp_insuf`` y ``artif_vent`` ya no se tratan como
   variables numéricas perturbables; se excluyen explícitamente de
   NUMERIC_COLS.
5. ``percent`` se excluye de la perturbación numérica por ser un
   parámetro de configuración, no una variable clínica.

Modo stub
---------
Si el proyecto SimUci no está disponible en el PYTHONPATH (por ejemplo, al
ejecutar las pruebas unitarias de este framework de forma aislada), el
adaptador activa automáticamente un modo stub con predicciones sintéticas
estadísticamente coherentes, para no bloquear el desarrollo ni las pruebas.

Autor   : Jorge Jesús Santos García
Tutora  : DrC. Deborah Raquel Galpert Cañizares
Centro  : UCLV — Departamento de Ciencias de la Computación
Año     : 2025
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ── Intento de importar el núcleo real de SimUci ────────────────────────────
# Las funciones reales viven en utils.helpers, no en un paquete "simuci"
# de nivel superior. Este import solo funcionará cuando este archivo se
# ejecute desde dentro de la raíz del proyecto SimUci-Streamlit (es decir,
# tras copiar simuci_framework/ a dicha raíz, tal como se hizo aquí).
try:
    from utils.helpers import (
        run_experiment as _run_experiment_real,
        predict as _predict_real,
        get_data_for_prediction as _get_data_for_prediction_real,
    )
    from utils.constants.paths import (
        FICHERODEDATOS_CSV_PATH as _REAL_CSV_PATH,
        PREDICTION_MODEL_PATH as _REAL_MODEL_PATH,
    )
    from utils.constants.limits import (
        AGE_MIN, AGE_MAX,
        APACHE_MIN, APACHE_MAX,
        VAM_T_MIN, VAM_T_MAX,
        UTI_STAY_MIN, UTI_STAY_MAX,
        PREUTI_STAY_MIN, PREUTI_STAY_MAX,
        SIM_RUNS_DEFAULT,
        SIM_PERCENT_DEFAULT,
    )
    _SIMUCI_AVAILABLE = True
except ImportError:
    _SIMUCI_AVAILABLE = False
    warnings.warn(
        "No se pudo importar utils.helpers / utils.constants del proyecto "
        "SimUci. El adaptador usará el modo stub (predicciones sintéticas). "
        "Verifica que simuci_framework/ esté en la raíz del proyecto "
        "SimUci-Streamlit y que el PYTHONPATH incluya dicha raíz.",
        ImportWarning,
        stacklevel=2,
    )
    # Valores de reserva idénticos a los reales, para que el modo stub
    # sea estadísticamente representativo incluso sin el proyecto real.
    AGE_MIN, AGE_MAX = 14, 100
    APACHE_MIN, APACHE_MAX = 0, 36
    VAM_T_MIN, VAM_T_MAX = 24, 700
    UTI_STAY_MIN, UTI_STAY_MAX = 0, 200
    PREUTI_STAY_MIN, PREUTI_STAY_MAX = 0, 34
    SIM_RUNS_DEFAULT = 200
    SIM_PERCENT_DEFAULT = 3

# ── Rutas del proyecto (usadas solo en modo stub / fallback) ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FALLBACK_CSV_PATH = _PROJECT_ROOT / "data" / "fichero_datos_MO_17_1_2023.csv"
_FALLBACK_MODEL_PATH = _PROJECT_ROOT / "models" / "prediction_model.joblib"

_CSV_PATH = _REAL_CSV_PATH if _SIMUCI_AVAILABLE else _FALLBACK_CSV_PATH
_MODEL_PATH = _REAL_MODEL_PATH if _SIMUCI_AVAILABLE else _FALLBACK_MODEL_PATH

# ── Columna objetivo del framework (literal, tal como la devuelve run_experiment) ──
TARGET_COL = "Estadia Post UCI"

# ── Variables NUMÉRICAS CONTINUAS del paciente, perturbables en stress-tests ──
# Excluye deliberadamente: n_runs y percent (parámetros de configuración de
# la simulación, no variables clínicas) y d1..d4/resp_insuf/artif_vent
# (códigos categóricos, no continuos).
NUMERIC_COLS = ["age", "apache", "vam_time", "uti_stay", "preuti_stay"]

# ── Variables categóricas / códigos (no perturbables como continuas) ────────
CATEGORICAL_COLS = ["d1", "d2", "d3", "d4", "resp_insuf", "artif_vent"]

# ── Parámetros de configuración de la simulación (nunca perturbables) ───────
SIMULATION_CONFIG_KEYS = {"n_runs", "percent"}

# ── Todas las features clínicas (orden usado por el modelo sustituto de regresión) ──
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# ── Rangos válidos por variable (para clamping defensivo, igual que hace SimUci) ──
VALID_RANGES = {
    "age": (AGE_MIN, AGE_MAX),
    "apache": (APACHE_MIN, APACHE_MAX),
    "vam_time": (VAM_T_MIN, VAM_T_MAX),
    "uti_stay": (UTI_STAY_MIN, UTI_STAY_MAX),
    "preuti_stay": (PREUTI_STAY_MIN, PREUTI_STAY_MAX),
}


# ─────────────────────────────────────────────────────────────────────────────
# Modo stub: simulación sintética estadísticamente coherente con los rangos reales
# ─────────────────────────────────────────────────────────────────────────────

class _StubSimulator:
    """
    Simulador sintético usado únicamente cuando el proyecto SimUci real no
    está disponible en el PYTHONPATH (p. ej. al ejecutar las pruebas
    unitarias del framework de forma aislada). Replica estadísticamente el
    comportamiento esperado usando los rangos reales verificados en
    utils/constants/limits.py.

    Nota de concurrencia: este simulador NO mantiene ningún generador
    aleatorio compartido a nivel de clase. Una versión anterior usaba un
    atributo de clase ``_rng`` mutado en cada llamada — seguro dentro de
    una única evaluación de un hilo, pero inseguro si múltiples sesiones
    de usuario de Streamlit (u otros hilos concurrentes del mismo proceso)
    invocan la simulación al mismo tiempo, ya que un único
    numpy.random.Generator no garantiza ser thread-safe para llamadas
    concurrentes sin sincronización externa. Cada llamada crea ahora su
    propio generador (semilla explícita para CRN, o semilla de entropía
    del sistema en caso contrario), eliminando el estado compartido entre
    llamadas o sesiones.
    """

    @classmethod
    def run_experiment_stub(
        cls,
        n_runs: int = SIM_RUNS_DEFAULT,
        age: int = 57,
        d1: int = 21, d2: int = 0, d3: int = 0, d4: int = 0,
        apache: int = 12,
        resp_insuf: int = 1,
        artif_vent: int = 0,
        vam_time: int = 120,
        uti_stay: int = 96,
        preuti_stay: int = 12,
        percent: int = SIM_PERCENT_DEFAULT,
        _crn_seed: int | None = None,
        **_kwargs,
    ) -> pd.DataFrame:
        """
        Genera un DataFrame sintético con las mismas columnas que run_experiment().

        Parameters
        ----------
        _crn_seed : int | None
            Semilla opcional para números aleatorios comunes (common random
            numbers, CRN — Law y Kelton, 2000). Cuando se proporciona, usa
            un generador local con esa semilla, permitiendo aparear la
            aleatoriedad interna entre una simulación base y una perturbada
            para que solo la diferencia en los parámetros de entrada
            explique la diferencia en la salida — cancelando el ruido de
            muestreo Monte Carlo en la comparación. Uso interno de
            RobustnessEvaluator; no disponible para el motor real de
            SimUci (su run_experiment() no expone un parámetro de semilla).
            Si no se proporciona, se usa un generador nuevo sembrado por
            entropía del sistema (sin estado compartido entre llamadas).
        """
        n_runs_safe = max(1, int(n_runs))
        rng = np.random.default_rng(_crn_seed)

        # Normalizar a escala relativa dentro de los rangos reales
        apache_norm = (apache - APACHE_MIN) / max(1, (APACHE_MAX - APACHE_MIN))
        vam_norm = (vam_time - VAM_T_MIN) / max(1, (VAM_T_MAX - VAM_T_MIN))
        uti_norm = (uti_stay - UTI_STAY_MIN) / max(1, (UTI_STAY_MAX - UTI_STAY_MIN))
        age_norm = (age - AGE_MIN) / max(1, (AGE_MAX - AGE_MIN))

        mu = 3.2 + 0.9 * apache_norm + 0.6 * vam_norm + 0.4 * uti_norm - 0.25 * age_norm
        sigma = 0.55 + 0.30 * apache_norm + 0.20 * age_norm

        pre_vam = rng.lognormal(mu - 0.8, sigma * 0.6, n_runs_safe)
        vam = rng.lognormal(mu, sigma, n_runs_safe)
        post_vam = rng.lognormal(mu - 0.4, sigma * 0.7, n_runs_safe)
        uci = rng.lognormal(mu + 0.3, sigma * 0.8, n_runs_safe)
        post_uci = rng.lognormal(mu, sigma, n_runs_safe)

        return pd.DataFrame({
            "Tiempo Pre VAM": pre_vam.clip(0),
            "Tiempo VAM": vam.clip(0),
            "Tiempo Post VAM": post_vam.clip(0),
            "Estadia UCI": uci.clip(0),
            "Estadia Post UCI": post_uci.clip(0),
        })

    @classmethod
    def predict_stub(
        cls,
        age: int = 57,
        d1: int = 21,
        apache: int = 12,
        vam_time: int = 120,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (clases, probabilidades) sintéticas para el modelo de clasificación."""
        apache_norm = (apache - APACHE_MIN) / max(1, (APACHE_MAX - APACHE_MIN))
        prob = min(0.95, max(0.02, 0.15 + 0.55 * apache_norm))
        cls_ = np.array([1 if prob > 0.5 else 0])
        prb = np.array([round(prob, 2)])
        return cls_, prb

    @classmethod
    def build_cohort(cls) -> pd.DataFrame:
        """Genera una cohorte sintética de 208 pacientes con rangos reales."""
        rng = np.random.default_rng(0)
        n = 208
        age = rng.normal(57.3, 18.4, n).clip(AGE_MIN, AGE_MAX).astype(int)
        apache = rng.normal(13.5, 5.5, n).clip(APACHE_MIN, APACHE_MAX).astype(int)
        vam_time = rng.lognormal(4.8, 0.85, n).clip(VAM_T_MIN, VAM_T_MAX).astype(int)
        uti_stay = rng.lognormal(4.5, 0.72, n).clip(UTI_STAY_MIN, UTI_STAY_MAX).astype(int)
        preuti_stay = rng.exponential(8.0, n).clip(PREUTI_STAY_MIN, PREUTI_STAY_MAX).astype(int)
        diag = rng.integers(1, 41, n)  # códigos PREUCI_DIAG 1-40

        df = pd.DataFrame({
            "age": age,
            "apache": apache,
            "vam_time": vam_time,
            "uti_stay": uti_stay,
            "preuti_stay": preuti_stay,
            "d1": diag,
            "d2": np.zeros(n, dtype=int),
            "d3": np.zeros(n, dtype=int),
            "d4": np.zeros(n, dtype=int),
            "resp_insuf": rng.integers(0, 6, n),
            "artif_vent": rng.integers(0, 3, n),
        })

        apache_norm = (df["apache"] - APACHE_MIN) / max(1, (APACHE_MAX - APACHE_MIN))
        vam_norm = (df["vam_time"] - VAM_T_MIN) / max(1, (VAM_T_MAX - VAM_T_MIN))
        mu = 3.2 + 0.9 * apache_norm + 0.6 * vam_norm
        sigma = 0.55 + 0.30 * apache_norm
        df[TARGET_COL] = rng.lognormal(mu, sigma)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Adaptador principal
# ─────────────────────────────────────────────────────────────────────────────

class SimUciAdapter:
    """
    Puente entre el framework de incertidumbre/robustez y el núcleo real de
    SimUci-Streamlit.

    Envuelve ``utils.helpers.run_experiment()`` (motor de simulación) y
    ``utils.helpers.predict()`` (modelo de clasificación de mortalidad),
    ambos verificados contra el código fuente real del proyecto. Opera en
    modo stub cuando esas funciones no están disponibles en el entorno.

    Parameters
    ----------
    csv_path : Path | str | None
        Ruta al CSV real de la cohorte. Por defecto usa la ruta oficial
        del proyecto (FICHERODEDATOS_CSV_PATH).
    calibration_split : float
        Fracción de la cohorte reservada para calibración conformal
        (default: 0.30, es decir 30% ~ 62 pacientes de 208).
    seed : int
        Semilla para la división entrenamiento/calibración/prueba.
    test_split : float
        Fracción reservada para un conjunto de PRUEBA independiente, que no
        interviene ni en el entrenamiento ni en la calibración (default: 0.0,
        sin conjunto de prueba, comportamiento original). Con
        ``calibration_split=0.20, test_split=0.20`` y n=208 resultan
        126/41/41 pacientes (entrenamiento/calibración/prueba).

    Examples
    --------
    >>> adapter = SimUciAdapter()
    >>> adapter.configure(dict(n_runs=200, age=57, apache=18,
    ...                        d1=21, d2=0, d3=0, d4=0,
    ...                        resp_insuf=1, artif_vent=0,
    ...                        vam_time=120, uti_stay=96,
    ...                        preuti_stay=12, percent=3))
    >>> pred = adapter.simulate()
    >>> X_cal, y_cal, model = adapter.get_conformal_inputs()
    """

    def __init__(
        self,
        csv_path: Path | str | None = None,
        calibration_split: float = 0.30,
        seed: int = 42,
        test_split: float = 0.0,
    ):
        self._csv_path = Path(csv_path) if csv_path else _CSV_PATH
        self._calibration_split = calibration_split
        self._test_split = test_split
        self._seed = seed
        self._stub_mode = not _SIMUCI_AVAILABLE

        self._patient_params: dict = {}
        self._is_configured = False
        self._cohort: pd.DataFrame | None = None
        self._cal_df: pd.DataFrame | None = None
        self._train_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None
        self._model: Any = None

        self._load_cohort()

    # ── Carga de la cohorte ──────────────────────────────────────────────────

    def _load_cohort(self):
        """Carga el CSV real (via Google Drive o local) o genera la cohorte sintética."""
        if not self._stub_mode:
            try:
                self._cohort = self._load_real_cohort()
            except Exception as exc:
                warnings.warn(
                    f"No se pudo cargar la cohorte real desde el proyecto SimUci "
                    f"({exc}). Usando cohorte sintética."
                )
                self._cohort = _StubSimulator.build_cohort()
        elif self._csv_path.exists():
            try:
                self._cohort = self._parse_local_csv(self._csv_path)
            except Exception as exc:
                warnings.warn(f"Error al parsear CSV local: {exc}. Usando cohorte sintética.")
                self._cohort = _StubSimulator.build_cohort()
        else:
            warnings.warn(
                f"CSV no encontrado en {self._csv_path}. Usando cohorte sintética."
            )
            self._cohort = _StubSimulator.build_cohort()

        self._split_cohort()

    def _load_real_cohort(self) -> pd.DataFrame:
        """
        Carga la cohorte real utilizando la infraestructura propia de SimUci
        (utils.data_loader.load_csv_from_drive, que resuelve tanto archivo
        local como descarga desde Google Drive según GOOGLE_DRIVE_FILE_MAP).
        """
        from utils.data_loader import load_csv_from_drive

        df = load_csv_from_drive("fichero_datos")
        return self._parse_dataframe(df)

    def _parse_local_csv(self, path: Path) -> pd.DataFrame:
        """Fallback: lee el CSV directamente desde disco (modo stub sin SimUci)."""
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        return self._parse_dataframe(df)

    def _parse_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte el CSV crudo de SimUci (columnas EXPERIMENT_VARIABLES_FROM_CSV:
        Edad, Diag.Ing1..4, APACHE, InsufResp, VA, Est. UCI, TiempoVAM,
        Est. PreUCI) al formato interno del framework.

        Nota: Est. UCI y Est. PreUCI vienen en DÍAS en el CSV original y se
        convierten a HORAS (x24) para ser consistentes con los rangos reales
        de run_experiment (UTI_STAY_MAX=200 h, PREUTI_STAY_MAX=34 h).
        """
        rename = {
            "Edad": "age",
            "APACHE": "apache",
            "TiempoVAM": "vam_time",
            "InsufResp": "resp_insuf",
            "VA": "artif_vent",
            "Diag.Ing1": "d1",
            "Diag.Ing2": "d2",
            "Diag.Ing3": "d3",
            "Diag.Ing4": "d4",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Est. UCI y Est. PreUCI: días -> horas
        if "Est. UCI" in df.columns:
            df["uti_stay"] = pd.to_numeric(df["Est. UCI"], errors="coerce").fillna(0) * 24
        if "Est. PreUCI" in df.columns:
            df["preuti_stay"] = pd.to_numeric(df["Est. PreUCI"], errors="coerce").fillna(0) * 24

        for col in ["age", "apache", "vam_time", "resp_insuf", "artif_vent",
                    "d1", "d2", "d3", "d4"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            else:
                df[col] = 0

        for col in ["uti_stay", "preuti_stay"]:
            if col not in df.columns:
                df[col] = 0

        # La cohorte cruda no trae la variable de salida objetivo
        # ("Estadia Post UCI" es resultado de la simulación, no un dato de
        # entrada); se completa ejecutando run_experiment de forma perezosa
        # solo cuando se necesite (ver get_conformal_inputs), para no
        # disparar cientos de simulaciones al simplemente cargar la cohorte.
        if TARGET_COL not in df.columns:
            df[TARGET_COL] = np.nan

        return df[ALL_FEATURES + [TARGET_COL]].copy()

    def _split_cohort(self):
        """
        Divide la cohorte en entrenamiento, calibración y (opcional) prueba.

        Orden de la permutación (semilla fija): primero calibración, luego
        prueba y el resto es entrenamiento. Con ``test_split=0`` se obtiene la
        misma partición entrenamiento/calibración que en la versión original.
        """
        if self._cohort is None:
            return
        rng = np.random.default_rng(self._seed)
        idx = rng.permutation(len(self._cohort))
        n_cal = int(len(self._cohort) * self._calibration_split)
        n_test = int(len(self._cohort) * self._test_split)
        self._cal_df = self._cohort.iloc[idx[:n_cal]].reset_index(drop=True)
        self._test_df = self._cohort.iloc[idx[n_cal:n_cal + n_test]].reset_index(drop=True)
        self._train_df = self._cohort.iloc[idx[n_cal + n_test:]].reset_index(drop=True)

    def _ensure_target_populated(self):
        """
        Completa la columna TARGET_COL ejecutando run_experiment() (o el
        stub) para las filas que aún no la tengan, con n_runs reducido para
        mantener el costo computacional acotado durante la calibración.
        """
        if self._cal_df is None:
            return
        missing = self._cal_df[TARGET_COL].isna()
        if not missing.any():
            return

        n_runs_calib = 50  # réplicas reducidas solo para poblar la calibración
        for idx in self._cal_df.index[missing]:
            row = self._cal_df.loc[idx]
            params = {col: int(row[col]) for col in ALL_FEATURES}
            params["n_runs"] = n_runs_calib
            params["percent"] = SIM_PERCENT_DEFAULT
            # Semilla determinista por fila (solo tiene efecto en modo stub)
            params["_crn_seed"] = self._seed * 100_000 + int(idx)
            try:
                pred = self.simulate(**params)
            except Exception:
                pred = float(self._cohort[TARGET_COL].dropna().mean()) if self._cohort[TARGET_COL].notna().any() else 96.0
            self._cal_df.loc[idx, TARGET_COL] = pred

    # ── Configuración por paciente ────────────────────────────────────────────

    def configure(self, patient_params: dict) -> "SimUciAdapter":
        """
        Configura el adaptador con los parámetros del paciente actual.

        Parameters
        ----------
        patient_params : dict
            Claves: n_runs, age, d1, d2, d3, d4, apache, resp_insuf,
                    artif_vent, vam_time, uti_stay, preuti_stay, percent.
        """
        self._patient_params = dict(patient_params)
        self._is_configured = True
        return self

    def clamp_patient_params(self, params: dict) -> dict:
        """
        Aplica el mismo clamping defensivo que usa SimUci internamente
        (ver utils.helpers.simulate_true_data.experiment_helper) para evitar
        que la simulación falle con valores fuera de rango.
        """
        clamped = dict(params)
        for key, (lo, hi) in VALID_RANGES.items():
            if key in clamped:
                clamped[key] = int(max(lo, min(clamped[key], hi)))
        return clamped

    # ── Función de simulación ─────────────────────────────────────────────────

    def simulate(self, **kwargs) -> float:
        """
        Ejecuta run_experiment() real (o el stub) y devuelve la media de
        "Estadia Post UCI".

        Esta es la función que reciben UncertaintyQuantifier y
        RobustnessEvaluator como ``simulate_fn``.
        """
        params = {**self._patient_params, **kwargs}
        params = self.clamp_patient_params(params)

        n_runs = int(params.get("n_runs", SIM_RUNS_DEFAULT))
        percent = int(params.get("percent", SIM_PERCENT_DEFAULT))

        if self._stub_mode:
            df = _StubSimulator.run_experiment_stub(**{**params, "n_runs": n_runs, "percent": percent})
            return float(df[TARGET_COL].mean())

        try:
            df = _run_experiment_real(
                n_runs=n_runs,
                age=int(params.get("age", 57)),
                d1=int(params.get("d1", 0)),
                d2=int(params.get("d2", 0)),
                d3=int(params.get("d3", 0)),
                d4=int(params.get("d4", 0)),
                apache=int(params.get("apache", 12)),
                resp_insuf=int(params.get("resp_insuf", 0)),
                artif_vent=int(params.get("artif_vent", 0)),
                vam_time=int(params.get("vam_time", VAM_T_MIN)),
                uti_stay=int(params.get("uti_stay", 24)),
                preuti_stay=int(params.get("preuti_stay", 0)),
                percent=percent,
            )
            if TARGET_COL in df.columns:
                return float(df[TARGET_COL].mean())
            return float(df.iloc[:, -1].mean())
        except Exception as exc:
            warnings.warn(f"run_experiment() real falló: {exc}. Usando stub.")
            df = _StubSimulator.run_experiment_stub(**{**params, "n_runs": n_runs, "percent": percent})
            return float(df[TARGET_COL].mean())

    def simulate_full(self, **kwargs) -> pd.DataFrame:
        """Ejecuta run_experiment() y devuelve el DataFrame completo (todas las fases)."""
        params = {**self._patient_params, **kwargs}
        params = self.clamp_patient_params(params)
        n_runs = int(params.get("n_runs", SIM_RUNS_DEFAULT))
        percent = int(params.get("percent", SIM_PERCENT_DEFAULT))

        if self._stub_mode:
            return _StubSimulator.run_experiment_stub(**{**params, "n_runs": n_runs, "percent": percent})

        try:
            return _run_experiment_real(
                n_runs=n_runs,
                age=int(params.get("age", 57)),
                d1=int(params.get("d1", 0)),
                d2=int(params.get("d2", 0)),
                d3=int(params.get("d3", 0)),
                d4=int(params.get("d4", 0)),
                apache=int(params.get("apache", 12)),
                resp_insuf=int(params.get("resp_insuf", 0)),
                artif_vent=int(params.get("artif_vent", 0)),
                vam_time=int(params.get("vam_time", VAM_T_MIN)),
                uti_stay=int(params.get("uti_stay", 24)),
                preuti_stay=int(params.get("preuti_stay", 0)),
                percent=percent,
            )
        except Exception as exc:
            warnings.warn(f"run_experiment() real falló: {exc}. Usando stub.")
            return _StubSimulator.run_experiment_stub(**{**params, "n_runs": n_runs, "percent": percent})

    def simulate_n(self, n: int = 20, **kwargs) -> list[float]:
        """
        Utilidad de conveniencia del framework (no forma parte de la API
        real de SimUci): ejecuta ``simulate()`` n veces de forma
        independiente y devuelve la lista de medias resultantes. Útil para
        diagnósticos rápidos de variabilidad fuera de los módulos formales
        de incertidumbre (UncertaintyQuantifier ya cubre este caso de uso
        con garantías estadísticas explícitas; este método es solo para
        inspección exploratoria).
        """
        return [self.simulate(**kwargs) for _ in range(n)]

    # ── Clasificación (modelo de mortalidad) ─────────────────────────────────

    def classify(self, patient_params: dict | None = None) -> dict:
        """
        Llama a predict() real del modelo de clasificación de SimUci
        (predice fallecimiento vs. no fallecimiento, NO el tiempo de
        supervivencia — son dos modelos distintos dentro de SimUci).

        Requiere: Diag.Ing1, Diag.Ing2, Diag.Egr2, TiempoVAM, APACHE, Edad.
        Diag.Egr2 (diagnóstico de EGRESO) no forma parte de los parámetros
        de entrada de este framework (que solo maneja variables de
        ingreso); se completa con 0 por defecto.

        Returns
        -------
        dict con claves: 'class' (0/1), 'probability', 'label'.
        """
        params = patient_params or self._patient_params
        params = self.clamp_patient_params(params)

        if self._stub_mode:
            cls_, prb = _StubSimulator.predict_stub(
                age=int(params.get("age", 57)),
                d1=int(params.get("d1", 0)),
                apache=int(params.get("apache", 12)),
                vam_time=int(params.get("vam_time", VAM_T_MIN)),
            )
        else:
            try:
                df_pred = _get_data_for_prediction_real({
                    "Edad": params.get("age", 57),
                    "Diag.Ing1": params.get("d1", 0),
                    "Diag.Ing2": params.get("d2", 0),
                    "Diag.Egr2": 0,   # no disponible en variables de ingreso
                    "TiempoVAM": params.get("vam_time", VAM_T_MIN),
                    "APACHE": params.get("apache", 12),
                })
                cls_, prb = _predict_real(df_pred)
            except Exception as exc:
                warnings.warn(f"predict() real falló: {exc}. Usando stub.")
                cls_, prb = _StubSimulator.predict_stub(
                    age=int(params.get("age", 57)),
                    apache=int(params.get("apache", 12)),
                )

        patient_lives = int(cls_[0]) == 0
        return {
            "class": int(cls_[0]),
            "probability": float(prb[0]),
            "label": (
                f"Paciente no fallece ({float(prb[0]) * 100:.0f}%)"
                if patient_lives
                else f"Paciente fallece ({float(prb[0]) * 100:.0f}%)"
            ),
        }

    # ── Datos para el framework ───────────────────────────────────────────────

    def get_calibration_data(self) -> tuple[pd.DataFrame, list[str]]:
        """Devuelve (DataFrame de calibración, columnas numéricas perturbables)."""
        if self._cal_df is None:
            self._load_cohort()
        return self._cal_df.copy(), NUMERIC_COLS  # type: ignore[return-value]

    def get_train_data(self) -> pd.DataFrame:
        """Devuelve el conjunto de entrenamiento (70% de la cohorte)."""
        if self._train_df is None:
            self._load_cohort()
        return self._train_df.copy()  # type: ignore[return-value]

    def get_test_data(self) -> pd.DataFrame:
        """
        Devuelve el conjunto de prueba (vacío si ``test_split=0``) con la
        variable de salida poblada mediante la simulación (n_runs reducido).
        """
        if self._test_df is None:
            self._load_cohort()
        df = self._test_df  # type: ignore[assignment]
        missing = df[TARGET_COL].isna()
        for idx in df.index[missing]:
            row = df.loc[idx]
            params = {col: int(row[col]) for col in ALL_FEATURES}
            params["n_runs"] = 50
            params["percent"] = SIM_PERCENT_DEFAULT
            params["_crn_seed"] = self._seed * 100_000 + 90_000 + int(idx)
            df.loc[idx, TARGET_COL] = self.simulate(**params)
        return df.copy()

    def get_conformal_inputs(self) -> tuple[NDArray, NDArray, Any]:
        """
        Devuelve (X_cal, y_cal, model) para inicializar ConformalPredictor.

        Si la cohorte real no trae precalculada la variable de salida
        (caso habitual, pues "Estadia Post UCI" es un resultado de
        simulación, no un dato crudo del CSV), esta función la completa
        ejecutando run_experiment() con n_runs reducido antes de entrenar
        el modelo sustituto de regresión.
        """
        if self._cal_df is None:
            self._load_cohort()

        self._ensure_target_populated()

        X_cal = self._cal_df[ALL_FEATURES].to_numpy(dtype=float)  # type: ignore
        y_cal = self._cal_df[TARGET_COL].to_numpy(dtype=float)  # type: ignore

        model = self._get_regression_model()
        return X_cal, y_cal, model

    def patient_feature_vector(self) -> NDArray:
        """Construye el vector de características del paciente configurado."""
        if not self._is_configured:
            raise RuntimeError(
                "Llama a .configure(patient_params) antes de patient_feature_vector()."
            )
        row = [float(self._patient_params.get(col, 0)) for col in ALL_FEATURES]
        return np.array(row, dtype=float).reshape(1, -1)

    # ── Modelo sustituto de regresión (para conformal prediction) ────────────

    def _get_regression_model(self) -> Any:
        """
        Random Forest de regresión (framework propio, distinto del modelo
        de clasificación real de SimUci) entrenado para predecir
        "Estadia Post UCI" a partir de ALL_FEATURES, usado como base para
        la conformal prediction del intervalo de supervivencia.
        """
        if self._model is not None:
            return self._model

        try:
            from sklearn.ensemble import RandomForestRegressor

            if self._train_df is None:
                self._load_cohort()

            train_df = self._train_df.copy()  # type: ignore
            if train_df[TARGET_COL].isna().any():
                # Poblar también el conjunto de entrenamiento si hace falta
                n_runs_calib = 50
                for idx in train_df.index[train_df[TARGET_COL].isna()]:
                    row = train_df.loc[idx]
                    params = {col: int(row[col]) for col in ALL_FEATURES}
                    params["n_runs"] = n_runs_calib
                    params["percent"] = SIM_PERCENT_DEFAULT
                    params["_crn_seed"] = self._seed * 100_000 + 50_000 + int(idx)
                    try:
                        train_df.loc[idx, TARGET_COL] = self.simulate(**params)
                    except Exception:
                        train_df.loc[idx, TARGET_COL] = 96.0

            X = train_df[ALL_FEATURES].to_numpy(dtype=float)
            y = train_df[TARGET_COL].to_numpy(dtype=float)

            self._model = RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                random_state=self._seed, n_jobs=-1,
            )
            self._model.fit(X, y)
            return self._model

        except ImportError:
            class _IdentityModel:
                def predict(self, X):
                    return np.full(len(X), 96.0)
            self._model = _IdentityModel()
            return self._model

    # ── Diagnóstico y propiedades ─────────────────────────────────────────────

    @property
    def stub_mode(self) -> bool:
        return self._stub_mode

    @property
    def cohort_size(self) -> int:
        return len(self._cohort) if self._cohort is not None else 0

    @property
    def calibration_size(self) -> int:
        return len(self._cal_df) if self._cal_df is not None else 0

    @property
    def test_size(self) -> int:
        return len(self._test_df) if self._test_df is not None else 0

    @property
    def train_size(self) -> int:
        return len(self._train_df) if self._train_df is not None else 0

    def __repr__(self) -> str:
        mode = "stub" if self._stub_mode else "SimUci-real"
        return (
            f"SimUciAdapter(mode={mode!r}, cohort={self.cohort_size}, "
            f"train={self.train_size}, calibration={self.calibration_size})"
        )
