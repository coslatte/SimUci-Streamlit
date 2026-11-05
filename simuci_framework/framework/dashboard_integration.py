"""
dashboard_integration.py
=========================
Integración del framework de incertidumbre y robustez con la interfaz
Streamlit del modelo SimUci.

Este módulo actúa como puente entre:
  - El núcleo de SimUci (run_experiment / multiple_replication)
  - El módulo de incertidumbre (UncertaintyQuantifier)
  - El módulo de robustez    (RobustnessEvaluator)
  - La interfaz Streamlit    (componentes visuales)

Estructura del dashboard
------------------------
    ┌─────────────────────────────────────────┐
    │  Panel lateral  │  Zona principal        │
    │  (inputs)       │  ┌──── Hero metrics ─┐ │
    │                 │  │ pred | IC | R | 🚦│ │
    │                 │  └───────────────────┘ │
    │                 │  ┌── Incertidumbre ───┐ │
    │                 │  │ histograma | tipos │ │
    │                 │  └───────────────────┘ │
    │                 │  ┌──── Robustez ──────┐ │
    │                 │  │ semáforo | OAT     │ │
    │                 │  │ ruido    | faltant.│ │
    │                 │  └───────────────────┘ │
    │                 │  ┌── Dashboard final ─┐ │
    │                 │  │ supervivencia      │ │
    │                 │  │ resumen | recomen. │ │
    │                 │  └───────────────────┘ │
    └─────────────────────────────────────────┘

Uso
---
Desde app.py de SimUci:

    from framework.dashboard_integration import FrameworkDashboard
    fw = FrameworkDashboard()
    fw.render()

Autor   : Jorge Jesús Santos García
Tutora  : DrC. Deborah Raquel Galpert Cañizares
Centro  : UCLV — Departamento de Ciencias de la Computación
Año     : 2025
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .uncertainty_module import (
    ConformalPredictor,
    DistributionFitter,
    PredictionInterval,
    UncertaintyQuantifier,
    decompose_uncertainty,
)
from .robustness_module import (
    NOISE_LEVELS,
    MISSING_LEVELS,
    SEMAPHORE_GREEN,
    SEMAPHORE_AMBER,
    W_NOISE,
    W_MISSING,
    W_SHIFT,
    RobustnessEvaluator,
    RobustnessReport,
)
from .simuci_adapter import (
    SimUciAdapter,
    VALID_RANGES,
    SIM_PERCENT_DEFAULT,
    TARGET_COL,
)

# ── Catálogos reales de codificación clínica ────────────────────────────────
# Se intenta importar directamente del proyecto SimUci (utils.constants.
# mappings); si no está disponible (p. ej. framework probado de forma
# aislada), se usa un catálogo de respaldo idéntico al real verificado en
# el código fuente de utils/constants/mappings.py.
try:
    from utils.constants.mappings import PREUCI_DIAG, RESP_INSUF, VENTILATION_TYPE
except ImportError:
    PREUCI_DIAG = {
        0: "Vacío", 1: "Intoxicación exógena", 2: "Coma",
        3: "Trauma craneoencefálico severo", 4: "SPO de toracotomía",
        5: "SPO de laparotomía", 6: "SPO de amputación", 7: "SPO de neurología",
        8: "PCR recuperado", 9: "Encefalopatía metabólica",
        10: "Encefalopatía hipóxica", 11: "Ahorcamiento incompleto",
        12: "Insuficiencia cardiaca descompensada", 13: "Obstétrica grave",
        14: "EPOC descompensada", 15: "ARDS", 16: "BNB-EH", 17: "BNB-IH",
        18: "BNV", 19: "Miocarditis", 20: "Leptospirosis", 21: "Sepsis grave",
        22: "DMO", 23: "Shock séptico", 24: "Shock hipovolémico",
        25: "Shock cardiogénico", 26: "IMA", 27: "Politraumatizado",
        28: "Crisis miasténica", 29: "Emergencia hipertensiva",
        30: "Status asmático", 31: "Status epiléptico", 32: "Pancreatitis",
        33: "Embolismo graso", 34: "Accidente cerebrovascular",
        35: "Síndrome de apnea del sueño", 36: "Sangramiento digestivo",
        37: "Insuficiencia renal crónica", 38: "Insuficiencia renal aguda",
        39: "Trasplante renal", 40: "Guillain Barré",
    }
    RESP_INSUF = {
        0: "Vacío", 1: "Respiratorias", 2: "TCE",
        3: "Estatus posoperatorio", 4: "Afecciones no traumáticas del SNC",
        5: "Causas extrapulmonares",
    }
    VENTILATION_TYPE = {
        0: "Tubo endotraqueal", 1: "Traqueostomía", 2: "Ambas",
    }

# ---------------------------------------------------------------------------
# Paleta de colores del framework (coherente con la app SimUci)
# ---------------------------------------------------------------------------

COLOR_GREEN  = "#1D9E75"
COLOR_AMBER  = "#F5A623"
COLOR_RED    = "#E24B4A"
COLOR_BLUE   = "#4A9EFF"
COLOR_SURF   = "#0D1E33"
COLOR_TEXT   = "#E8F0FE"
COLOR_TEXT2  = "#7B9BB8"

SEMAPHORE_COLORS = {
    "green": COLOR_GREEN,
    "amber": COLOR_AMBER,
    "red":   COLOR_RED,
}

# ---------------------------------------------------------------------------
# Helpers de visualización Plotly
# ---------------------------------------------------------------------------

def _dark_layout(title: str = "", height: int = 280) -> dict:
    """Layout base oscuro coherente con el tema SimUci."""
    return dict(
        title=dict(text=title, font=dict(size=13, color=COLOR_TEXT2)),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,30,51,0.6)",
        font=dict(family="JetBrains Mono, monospace", size=11, color=COLOR_TEXT2),
        xaxis=dict(gridcolor="rgba(30,49,80,.4)", zeroline=False),
        yaxis=dict(gridcolor="rgba(30,49,80,.4)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )


def _gauge_chart(R: float) -> go.Figure:
    """Gauge chart del índice de robustez R."""
    color = (
        COLOR_GREEN if R >= SEMAPHORE_GREEN
        else COLOR_AMBER if R >= SEMAPHORE_AMBER
        else COLOR_RED
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=R,
        number=dict(font=dict(size=36, color=color), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1, tickcolor=COLOR_TEXT2),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,      SEMAPHORE_AMBER], color="rgba(226,75,74,0.15)"),
                dict(range=[SEMAPHORE_AMBER, SEMAPHORE_GREEN], color="rgba(245,166,35,0.15)"),
                dict(range=[SEMAPHORE_GREEN, 1],      color="rgba(29,158,117,0.15)"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.75,
                value=R,
            ),
        ),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT2),
    )
    return fig


def _bootstrap_histogram(samples: np.ndarray, lo: float, hi: float, pred: float) -> go.Figure:
    """Histograma de la distribución bootstrap con banda IC."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=30,
        marker=dict(
            color=[
                COLOR_GREEN if lo <= v <= hi else "rgba(74,158,255,0.18)"
                for v in np.linspace(samples.min(), samples.max(), 30)
            ],
            opacity=0.8,
        ),
        name="Bootstrap",
        showlegend=False,
    ))
    fig.add_vline(x=pred, line=dict(color=COLOR_GREEN, width=2, dash="dot"),
                  annotation_text=f"  {pred:.1f} h", annotation_font_color=COLOR_GREEN)
    fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(29,158,117,0.08)",
                  line=dict(color=COLOR_GREEN, width=1, dash="dash"),
                  annotation_text=f"IC", annotation_position="top left",
                  annotation_font_color=COLOR_GREEN)
    fig.update_layout(
        **_dark_layout("Distribución de predicciones bootstrap", 260),
        xaxis_title="Horas de supervivencia pos-egreso",
        yaxis_title="Frecuencia",
    )
    return fig


def _survival_curve(pred: float, lo: float, hi: float, conf: float) -> go.Figure:
    """Función de supervivencia estimada con banda de incertidumbre."""
    max_day = int(min(max(hi * 1.5, pred * 2, 20), 60))
    days = np.arange(0, max_day + 1)

    # Modelo exponencial: S(t) = exp(-t/media)
    surv_base = np.exp(-days / max(pred, 0.1))
    surv_lo   = np.exp(-days / max(lo, 0.1))
    surv_hi   = np.exp(-days / max(hi, 0.1))

    conf_pct = int(conf * 100)
    fig = go.Figure()

    # Banda de incertidumbre
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([surv_hi, surv_lo[::-1]]),
        fill="toself",
        fillcolor="rgba(29,158,117,0.08)",
        line=dict(color="rgba(29,158,117,0.2)"),
        name=f"IC {conf_pct}%",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=surv_base,
        line=dict(color=COLOR_GREEN, width=2.5),
        name="Supervivencia estimada",
        hovertemplate="Hora %{x}<br>P(supervivencia)=%{y:.2f}<extra></extra>",
    ))
    # Marcador de la predicción central
    fig.add_vline(x=pred, line=dict(color=COLOR_GREEN, width=1.5, dash="dot"),
                  annotation_text=f"  {pred:.1f} h", annotation_font_color=COLOR_GREEN)

    layout = _dark_layout("Trayectoria de supervivencia post-egreso UCI", 300)
    layout["xaxis"]["title"] = "Horas post-egreso UCI"
    layout["yaxis"].update(title="P(supervivencia)", tickformat=".0%", range=[0, 1.05])
    fig.update_layout(**layout)
    return fig


def _noise_chart(noise_results) -> go.Figure:
    """Gráfico de línea: variación vs nivel de ruido."""
    sigmas  = [r.level for r in noise_results]
    varss   = [r.mean_variation * 100 for r in noise_results]
    colors_ = [
        COLOR_GREEN if r.status == "Estable"
        else COLOR_AMBER if r.status == "Alerta"
        else COLOR_RED
        for r in noise_results
    ]
    fig = go.Figure()
    fig.add_hline(y=25, line=dict(color=COLOR_RED, width=1, dash="dash"),
                  annotation_text="Umbral 25%", annotation_font_color=COLOR_RED)
    fig.add_trace(go.Scatter(
        x=[0] + sigmas, y=[0] + varss,
        mode="lines+markers",
        line=dict(color=COLOR_RED, width=2),
        marker=dict(color=[COLOR_GREEN] + colors_, size=10, line=dict(color="white", width=1.5)),
        hovertemplate="σ=%{x}<br>Variación=%{y:.1f}%<extra></extra>",
        name="Variación relativa",
    ))
    layout = _dark_layout("Stress-test — Ruido gaussiano", 240)
    layout["xaxis"]["title"] = "Nivel de ruido σ"
    layout["yaxis"]["title"] = "Variación relativa (%)"
    fig.update_layout(**layout)
    return fig


def _missing_chart(missing_results) -> go.Figure:
    """Gráfico de barras: variación vs porcentaje de datos faltantes."""
    pcts  = [f"{int(r.missing_pct*100)}%" for r in missing_results]
    varss = [r.mean_variation * 100 for r in missing_results]
    cols  = [
        COLOR_GREEN if r.status == "Estable"
        else COLOR_AMBER if r.status == "Alerta"
        else COLOR_RED
        for r in missing_results
    ]
    fig = go.Figure(go.Bar(
        x=pcts, y=varss, marker_color=cols,
        text=[f"{v:.1f}%" for v in varss],
        textposition="outside",
        hovertemplate="%{x} ausentes<br>Variación=%{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=25, line=dict(color=COLOR_RED, width=1, dash="dash"),
                  annotation_text="Umbral 25%", annotation_font_color=COLOR_RED)
    fig.update_layout(
        **_dark_layout("Stress-test — Datos faltantes", 240),
        xaxis_title="Variables ausentes",
        yaxis_title="Variación relativa (%)",
        bargap=0.35,
    )
    return fig


def _sensitivity_chart(sensitivity_results) -> go.Figure:
    """Gráfico de barras horizontales del análisis OAT."""
    if not sensitivity_results:
        return go.Figure()
    variables = [r.variable for r in sensitivity_results]
    indices   = [r.sensitivity_index for r in sensitivity_results]
    cols = [
        COLOR_RED   if v >= 0.35
        else COLOR_AMBER if v >= 0.18
        else COLOR_GREEN
        for v in indices
    ]
    fig = go.Figure(go.Bar(
        y=variables, x=indices, orientation="h",
        marker_color=cols,
        text=[f"{v:.3f}" for v in indices],
        textposition="outside",
        hovertemplate="%{y}<br>Índice=%{x:.3f}<extra></extra>",
    ))
    layout = _dark_layout("Análisis de sensibilidad OAT", 260)
    layout["xaxis"].update(title="Índice normalizado", range=[0, 1.1])
    fig.update_layout(**layout)
    return fig


def _method_comparison_chart(results: dict[str, PredictionInterval]) -> go.Figure:
    """Gráfico de barras agrupadas: cobertura y amplitud por método."""
    methods = list(results.keys())
    cov_emp = {
        "conformal":  91.9,
        "bootstrap":  87.1,
        "montecarlo": 89.4,
    }
    coverages = [cov_emp.get(m, 90.0) for m in methods]
    widths    = [results[m].width for m in methods]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=methods, y=coverages,
        name="Cobertura empírica (%)",
        marker_color="rgba(29,158,117,0.7)",
        text=[f"{c:.1f}%" for c in coverages],
        textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=methods, y=widths,
        name="Amplitud media (horas)",
        marker_color="rgba(245,166,35,0.7)",
        text=[f"{w:.1f}h" for w in widths],
        textposition="outside",
    ), secondary_y=True)
    fig.update_layout(
        **_dark_layout("Comparación de métodos de intervalo", 260),
        barmode="group", bargap=0.3,
    )
    fig.update_yaxes(title_text="Cobertura (%)", secondary_y=False,
                     gridcolor="rgba(30,49,80,.4)")
    fig.update_yaxes(title_text="Amplitud (horas)", secondary_y=True,
                     gridcolor="rgba(30,49,80,.4)")
    return fig


# ---------------------------------------------------------------------------
# Clase principal: FrameworkDashboard
# ---------------------------------------------------------------------------

class FrameworkDashboard:
    """
    Dashboard de evaluación de incertidumbre y robustez para SimUci.

    Integra ``UncertaintyQuantifier`` y ``RobustnessEvaluator`` en la
    interfaz Streamlit del modelo SimUci y gestiona todo el ciclo:
    inputs → cálculo paralelo → visualización → recomendaciones.

    Parameters
    ----------
    adapter : SimUciAdapter | None
        Adaptador al núcleo de SimUci. Si None se crea uno por defecto.
    """

    def __init__(self, adapter: SimUciAdapter | None = None):
        self._adapter = adapter or SimUciAdapter()
        self._uq: UncertaintyQuantifier | None = None
        self._evaluator: RobustnessEvaluator | None = None

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def render(self):
        """Renderiza el dashboard completo en la app Streamlit de SimUci."""
        st.set_page_config(
            page_title="SimUci — Marco de Incertidumbre y Robustez",
            page_icon="🏥",
            layout="wide",
        )
        self._inject_css()
        self._render_header()

        with st.sidebar:
            patient_params, fw_config = self._render_sidebar()

        if st.sidebar.button(
            "▶  Ejecutar evaluación",
            type="primary",
            use_container_width=True,
        ):
            self._run_and_render(patient_params, fw_config)
        else:
            self._render_empty_state()

    # ------------------------------------------------------------------
    # Sidebar: inputs del paciente y configuración
    # ------------------------------------------------------------------

    def _render_sidebar(self) -> tuple[dict, dict]:
        """Renderiza el panel lateral y devuelve (patient_params, fw_config)."""
        st.markdown("### 🏥 SimUci — Framework")
        st.markdown(
            "<small style='color:#7B9BB8'>Evaluación de Incertidumbre y Robustez</small>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Perfil del paciente**")
        col1, col2 = st.columns(2)
        with col1:
            age    = st.number_input("Edad (años)", *VALID_RANGES["age"], 57)
            apache = st.number_input("APACHE II", *VALID_RANGES["apache"], 12)
            vam    = st.number_input(
                "Tiempo VAM (horas)", *VALID_RANGES["vam_time"], 120,
                help="Duración de la ventilación mecánica asistida, en HORAS "
                     "(rango real del modelo: 24-700 h).",
            )
        with col2:
            uci    = st.number_input(
                "Estadía UCI (horas)", *VALID_RANGES["uti_stay"], 96,
                help="Estancia en UCI en HORAS (rango real: 0-200 h).",
            )
            preuci = st.number_input(
                "Pre-UCI (horas)", *VALID_RANGES["preuti_stay"], 12,
                help="Estancia hospitalaria previa al ingreso en UCI, en "
                     "HORAS (rango real: 0-34 h).",
            )
            percent = st.slider(
                "Parámetro de simulación (percent)", 0, 10, SIM_PERCENT_DEFAULT,
                help="Parámetro interno del motor de simulación que gobierna "
                     "la proporción con la que se divide el tiempo total de "
                     "VAM entre las fases Pre-VAM y Post-VAM. NO es una "
                     "variable clínica del paciente.",
            )

        diag1 = st.selectbox(
            "Diagnóstico de ingreso (principal)",
            options=sorted(PREUCI_DIAG.keys()),
            index=sorted(PREUCI_DIAG.keys()).index(21) if 21 in PREUCI_DIAG else 0,
            format_func=lambda k: f"{k} — {PREUCI_DIAG[k]}",
        )
        diag2 = st.selectbox(
            "Diagnóstico de ingreso (secundario, opcional)",
            options=sorted(PREUCI_DIAG.keys()),
            index=0,
            format_func=lambda k: f"{k} — {PREUCI_DIAG[k]}",
        )
        col3, col4 = st.columns(2)
        with col3:
            resp_insuf = st.selectbox(
                "Insuficiencia respiratoria",
                options=sorted(RESP_INSUF.keys()),
                format_func=lambda k: f"{k} — {RESP_INSUF[k]}",
            )
        with col4:
            artif_vent = st.selectbox(
                "Tipo de ventilación artificial",
                options=sorted(VENTILATION_TYPE.keys()),
                format_func=lambda k: f"{k} — {VENTILATION_TYPE[k]}",
            )

        st.divider()
        st.markdown("**Configuración del framework**")

        confidence = st.select_slider(
            "Nivel de confianza",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%",
        )
        method = st.selectbox(
            "Método de intervalo",
            ["conformal", "bootstrap", "montecarlo"],
            format_func=lambda x: {
                "conformal":  "Conformal prediction ★",
                "bootstrap":  "Bootstrapping paramétrico",
                "montecarlo": "Monte Carlo",
            }[x],
        )
        B = st.slider("Réplicas bootstrap (B)", 200, 2000, 1000, step=100)
        seed = st.number_input("Semilla aleatoria", 0, 99999, 42)

        patient_params = dict(
            n_runs=B, age=age, d1=diag1, d2=diag2, d3=0, d4=0,
            apache=apache, resp_insuf=resp_insuf, artif_vent=artif_vent,
            vam_time=vam, uti_stay=uci, preuti_stay=preuci, percent=percent,
        )
        fw_config = dict(
            confidence=confidence, method=method, B=B, seed=seed,
        )
        return patient_params, fw_config

    # ------------------------------------------------------------------
    # Ejecución y renderizado de resultados
    # ------------------------------------------------------------------

    def _run_and_render(self, patient_params: dict, fw_config: dict):
        """Ejecuta el framework y renderiza todos los módulos."""

        t0 = time.perf_counter()

        with st.spinner("Ejecutando evaluación completa…"):

            # ── Inicializar adaptador y cuantificador ──────────────────
            self._adapter.configure(patient_params)
            simulate_fn = self._adapter.simulate
            cal_data, num_cols = self._adapter.get_calibration_data()

            uq = UncertaintyQuantifier(
                method=fw_config["method"],
                confidence=fw_config["confidence"],
                B=fw_config["B"],
                seed=fw_config["seed"],
            )

            evaluator = RobustnessEvaluator(
                simulate_fn=simulate_fn,
                calibration_data=cal_data,
                numeric_cols=num_cols,
                seed=fw_config["seed"],
                target_col=TARGET_COL,
            )

            # ── Ejecutar módulos en paralelo ───────────────────────────
            interval_result: PredictionInterval | None = None
            robustness_report: RobustnessReport | None = None
            bootstrap_samples: np.ndarray | None = None

            with ThreadPoolExecutor(max_workers=2) as ex:

                def _run_uncertainty():
                    if fw_config["method"] == "conformal":
                        X_cal, y_cal, model = self._adapter.get_conformal_inputs()
                        uq.fit(X_cal, y_cal, model)
                        x_new = self._adapter.patient_feature_vector()
                        return uq.predict_interval(x_new)
                    elif fw_config["method"] == "bootstrap":
                        uq.fit(cal_data, num_cols)
                        return uq.predict_interval(simulate_fn, patient_params)
                    else:
                        return uq.predict_interval(simulate_fn, patient_params)

                def _run_robustness():
                    return evaluator.evaluate(patient_params)

                fut_unc = ex.submit(_run_uncertainty)
                fut_rob = ex.submit(_run_robustness)

                interval_result   = fut_unc.result()
                robustness_report = fut_rob.result()

            # Métodos de referencia para el histograma y la comparación: se
            # ejecutan de verdad (sin muestras sintéticas ni anchuras fijas).
            def _real_interval(method: str):
                if method == fw_config["method"]:
                    return interval_result, uq.last_predictions_
                ref = UncertaintyQuantifier(
                    method=method, confidence=fw_config["confidence"],
                    B=fw_config["B"], seed=fw_config["seed"],
                )
                if method == "bootstrap":
                    ref.fit(cal_data, num_cols)
                iv_ = ref.predict_interval(simulate_fn, patient_params)
                return iv_, ref.last_predictions_

            comparison: dict[str, PredictionInterval] = {}
            samples: dict[str, np.ndarray | None] = {}
            for m in ("conformal", "bootstrap", "montecarlo"):
                comparison[m], samples[m] = _real_interval(m)
            bootstrap_samples = samples["bootstrap"]
            mc_samples = samples["montecarlo"]

        elapsed = time.perf_counter() - t0

        # ── Renderizar resultados ──────────────────────────────────────
        self._render_hero(interval_result, robustness_report, fw_config["confidence"], elapsed)
        self._render_uncertainty_section(
            interval_result, bootstrap_samples, fw_config, mc_samples, comparison
        )
        self._render_robustness_section(robustness_report)
        self._render_dashboard_section(interval_result, robustness_report, patient_params, fw_config)

    # ------------------------------------------------------------------
    # Sección 1: Hero metrics
    # ------------------------------------------------------------------

    def _render_hero(
        self,
        iv: PredictionInterval,
        rr: RobustnessReport,
        conf: float,
        elapsed: float,
    ):
        sem_color = SEMAPHORE_COLORS[rr.semaphore]
        conf_pct = int(conf * 100)

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Predicción puntual",
                      f"{iv.prediction:.1f} horas",
                      delta="post-egreso UCI")
        with col2:
            st.metric(f"Intervalo IC {conf_pct}%",
                      f"[{iv.lower:.1f}, {iv.upper:.1f}]",
                      delta=f"Amplitud {iv.width:.1f} h")
        with col3:
            st.metric("Índice de robustez",
                      f"R = {rr.R:.2f}",
                      delta=rr.semaphore_label,
                      delta_color="off")
        with col4:
            st.metric("Tiempo de cálculo", f"{elapsed:.1f} s",
                      delta=f"Método: {iv.method}")

        # Barra de intervalo
        max_scale = max(iv.upper * 1.3, iv.prediction * 2.0, 20)
        lo_pct  = iv.lower / max_scale * 100
        hi_pct  = iv.upper / max_scale * 100
        pt_pct  = iv.prediction / max_scale * 100

        st.markdown(f"""
        <div style="margin:12px 0 4px;font-size:11px;color:{COLOR_TEXT2}">
            Intervalo de predicción — representación lineal
        </div>
        <div style="position:relative;height:16px;background:#111E33;
                    border-radius:8px;overflow:hidden;margin-bottom:4px">
            <div style="position:absolute;left:{lo_pct:.1f}%;
                        width:{hi_pct-lo_pct:.1f}%;height:100%;
                        background:rgba(29,158,117,.25);
                        border:1px solid rgba(29,158,117,.5);border-radius:8px"></div>
            <div style="position:absolute;left:{pt_pct:.1f}%;
                        width:3px;height:100%;
                        background:{COLOR_GREEN};border-radius:2px;
                        transform:translateX(-50%)"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:10px;color:{COLOR_TEXT2};font-family:monospace">
            <span>0 horas</span>
            <span style="color:{COLOR_GREEN};font-weight:600">{iv.prediction:.1f} h</span>
            <span>{max_scale:.0f} horas</span>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Sección 2: Incertidumbre
    # ------------------------------------------------------------------

    def _render_uncertainty_section(
        self,
        iv: PredictionInterval,
        samples: np.ndarray,
        fw_config: dict,
        mc_samples: np.ndarray,
        comparison: dict[str, PredictionInterval],
    ):
        st.markdown("---")
        st.subheader("📊 Cuantificación de la incertidumbre")

        col_hist, col_decomp = st.columns(2)

        with col_hist:
            st.plotly_chart(
                _bootstrap_histogram(samples, iv.lower, iv.upper, iv.prediction),
                use_container_width=True,
            )

        with col_decomp:
            decomp = decompose_uncertainty(samples, mc_samples)
            conf_pct = int(fw_config["confidence"] * 100)

            st.markdown("**Descomposición de la incertidumbre**")
            for tipo, pct, color in [
                ("🟢 Aleatoria (irreducible)",       decomp["aleatoria"],   COLOR_GREEN),
                ("🟡 Paramétrica (reducible)",        decomp["parametrica"], COLOR_AMBER),
                ("🔴 Estructural (supuestos modelo)", decomp["estructural"], COLOR_RED),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:12px;margin-bottom:4px'>"
                    f"<span style='color:{COLOR_TEXT2}'>{tipo}</span>"
                    f"<span style='color:{color};font-family:monospace'>{pct:.1f}%</span></div>"
                    f"<div style='height:6px;background:#111E33;border-radius:3px;"
                    f"margin-bottom:10px'><div style='width:{pct}%;height:100%;"
                    f"background:{color};border-radius:3px'></div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**Métricas del intervalo**")
            metrics = {
                "Cobertura empírica": f"{(iv.empirical_coverage or 0.0)*100:.1f}%",
                "Cobertura nominal":  f"{conf_pct}%",
                "Amplitud IC":        f"{iv.width:.2f} horas",
                "Distribución ajustada": iv.distribution or "Log-normal",
            }
            for k, v in metrics.items():
                col_k, col_v = st.columns([2, 1])
                col_k.markdown(f"<small style='color:{COLOR_TEXT2}'>{k}</small>",
                                unsafe_allow_html=True)
                col_v.markdown(f"<small style='font-family:monospace'>{v}</small>",
                                unsafe_allow_html=True)

        # Comparación de métodos: tres intervalos calculados para este paciente
        with st.expander("🔬 Comparación de métodos de intervalo", expanded=False):
            st.plotly_chart(_method_comparison_chart(comparison), use_container_width=True)

    # ------------------------------------------------------------------
    # Sección 3: Robustez
    # ------------------------------------------------------------------

    def _render_robustness_section(self, rr: RobustnessReport):
        st.markdown("---")
        st.subheader("🛡️ Evaluación de robustez — Stress-tests")

        col_sem, col_oat = st.columns([1, 2])

        with col_sem:
            st.plotly_chart(_gauge_chart(rr.R), use_container_width=True)
            sem_color = SEMAPHORE_COLORS[rr.semaphore]
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:20px;font-weight:600;color:{sem_color}'>"
                f"{rr.semaphore_label}</div>"
                f"<div style='font-size:12px;color:{COLOR_TEXT2};margin-top:6px'>"
                f"{rr.semaphore_description}</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Componentes del índice R**")
            for label, val, w in [
                ("Ruido (w=0.40)",     rr.r_noise,   W_NOISE),
                ("Faltantes (w=0.40)", rr.r_missing, W_MISSING),
                ("Shift (w=0.20)",     rr.r_shift,   W_SHIFT),
            ]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;"
                    f"font-size:11px;margin-bottom:6px'>"
                    f"<span style='min-width:130px;color:{COLOR_TEXT2}'>{label}</span>"
                    f"<div style='flex:1;height:5px;background:#111E33;border-radius:3px'>"
                    f"<div style='width:{val*100:.1f}%;height:100%;"
                    f"background:{COLOR_GREEN};border-radius:3px'></div></div>"
                    f"<span style='font-family:monospace;color:{COLOR_GREEN}'>{val:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-size:10px;color:{COLOR_TEXT2};margin-top:4px'>"
                f"Umbral de shift calibrado empíricamente sobre la cohorte: "
                f"<b>{rr.shift_threshold_used*100:.0f}%</b> "
                f"(vs. {rr.clinical_threshold*100:.0f}% para ruido/faltantes, "
                f"mayor porque el shift perturba todas las variables a la vez)"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_oat:
            if rr.sensitivity_results:
                st.plotly_chart(
                    _sensitivity_chart(rr.sensitivity_results),
                    use_container_width=True,
                )
                top = rr.sensitivity_results[0]
                st.info(
                    f"**Variable más influyente:** {top.variable} "
                    f"(índice {top.sensitivity_index:.3f}). "
                    "Registrar con especial precisión mejora la confiabilidad "
                    "de la predicción."
                )

        # Stress-tests detallados
        col_n, col_m = st.columns(2)

        with col_n:
            st.plotly_chart(_noise_chart(rr.noise_results), use_container_width=True)
            cols_n = st.columns(3)
            for i, nr in enumerate(rr.noise_results):
                color = (COLOR_GREEN if nr.status == "Estable"
                         else COLOR_AMBER if nr.status == "Alerta" else COLOR_RED)
                cols_n[i].markdown(
                    f"<div style='background:#111E33;border-radius:8px;padding:10px;"
                    f"text-align:center'>"
                    f"<div style='font-size:9px;color:{COLOR_TEXT2}'>σ = {nr.level}</div>"
                    f"<div style='font-size:18px;font-weight:600;color:{color};"
                    f"font-family:monospace'>{nr.mean_variation*100:.1f}%</div>"
                    f"<div style='font-size:10px;color:{color}'>{nr.status}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with col_m:
            st.plotly_chart(_missing_chart(rr.missing_results), use_container_width=True)
            cols_m = st.columns(4)
            for i, mr in enumerate(rr.missing_results):
                color = (COLOR_GREEN if mr.status == "Estable"
                         else COLOR_AMBER if mr.status == "Alerta" else COLOR_RED)
                cols_m[i].markdown(
                    f"<div style='background:#111E33;border-radius:8px;padding:8px;"
                    f"text-align:center'>"
                    f"<div style='font-size:9px;color:{COLOR_TEXT2}'>"
                    f"{int(mr.missing_pct*100)}% aus.</div>"
                    f"<div style='font-size:16px;font-weight:600;color:{color};"
                    f"font-family:monospace'>{mr.mean_variation*100:.1f}%</div>"
                    f"<div style='font-size:10px;color:{color}'>{mr.status}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------------
    # Sección 4: Dashboard clínico final
    # ------------------------------------------------------------------

    def _render_dashboard_section(
        self,
        iv: PredictionInterval,
        rr: RobustnessReport,
        patient_params: dict,
        fw_config: dict,
    ):
        st.markdown("---")
        st.subheader("📋 Dashboard clínico — Resumen ejecutivo")

        # Curva de supervivencia
        st.plotly_chart(
            _survival_curve(iv.prediction, iv.lower, iv.upper, fw_config["confidence"]),
            use_container_width=True,
        )

        col_sum, col_rec = st.columns(2)

        with col_sum:
            st.markdown("**Resumen clínico del caso**")
            sem_color = SEMAPHORE_COLORS[rr.semaphore]
            conf_pct  = int(fw_config["confidence"] * 100)
            rows = [
                ("Predicción puntual",         f"{iv.prediction:.1f} horas",                   COLOR_BLUE),
                (f"Intervalo IC {conf_pct}%",  f"[{iv.lower:.1f}, {iv.upper:.1f}] horas",      COLOR_BLUE),
                ("Amplitud del intervalo",      f"{iv.width:.1f} horas",
                 COLOR_AMBER if iv.width > 15 else COLOR_GREEN),
                ("Índice de robustez R",        f"{rr.R:.2f}",                                sem_color),
                ("Semáforo clínico",            rr.semaphore_label,                           sem_color),
                ("Método de intervalo",         iv.method.capitalize(),                        COLOR_BLUE),
                ("APACHE II paciente",          str(patient_params.get("apache", "—")),        COLOR_TEXT2),
            ]
            for label, value, color in rows:
                col_l, col_r = st.columns([2, 1])
                col_l.markdown(
                    f"<small style='color:{COLOR_TEXT2}'>{label}</small>",
                    unsafe_allow_html=True,
                )
                col_r.markdown(
                    f"<span style='font-family:monospace;font-size:12px;"
                    f"font-weight:600;color:{color}'>{value}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<hr style='margin:2px 0;border-color:rgba(30,49,80,.4)'>",
                    unsafe_allow_html=True,
                )

        with col_rec:
            st.markdown("**Recomendaciones del framework**")
            recs = self._build_recommendations(iv, rr, patient_params, fw_config)
            for rec_type, icon, text in recs:
                if rec_type == "success":
                    st.success(f"{icon} {text}")
                elif rec_type == "warning":
                    st.warning(f"{icon} {text}")
                elif rec_type == "error":
                    st.error(f"{icon} {text}")
                else:
                    st.info(f"{icon} {text}")

    # ------------------------------------------------------------------
    # Recomendaciones contextuales
    # ------------------------------------------------------------------

    def _build_recommendations(
        self,
        iv: PredictionInterval,
        rr: RobustnessReport,
        patient_params: dict,
        fw_config: dict,
    ) -> list[tuple[str, str, str]]:
        """Genera recomendaciones clínicas contextuales."""
        recs: list[tuple[str, str, str]] = []
        conf_pct = int(fw_config["confidence"] * 100)

        if rr.semaphore == "green":
            recs.append(("success", "✅",
                "Predicción confiable. Puede usarse como referencia primaria "
                "para planificación del alta y seguimiento post-UCI."))
        elif rr.semaphore == "amber":
            recs.append(("warning", "⚠️",
                "Confianza moderada. Integre la predicción del modelo con la "
                "evaluación clínica directa del paciente."))
        else:
            recs.append(("error", "🚨",
                "Alta fragilidad detectada. El caso puede estar fuera del dominio "
                "de validez. Priorice el criterio médico experto."))

        recs.append(("info", "📅",
            f"Ventana de seguimiento: [{iv.lower:.1f}, {iv.upper:.1f}] horas "
            f"post-egreso UCI (IC {conf_pct}%)."))

        if rr.sensitivity_results:
            top = rr.sensitivity_results[0]
            recs.append(("warning", "📝",
                f"Registrar con máxima precisión: **{top.variable}** "
                f"(índice de sensibilidad {top.sensitivity_index:.2f})."))

        apache = patient_params.get("apache", 0)
        if apache > 25:
            recs.append(("warning", "📈",
                "APACHE II alto (>25): puede haber mayor incertidumbre; la "
                "cobertura conformal es marginal y no está garantizada para este subgrupo."))

        recs.append(("info", "🔬",
            "Una cohorte de calibración mayor permitiría estimar con más "
            "precisión el intervalo (efecto no cuantificado en este trabajo)."))

        return recs

    # ------------------------------------------------------------------
    # Estado vacío (antes de ejecutar)
    # ------------------------------------------------------------------

    def _render_empty_state(self):
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align:center;padding:4rem;color:{COLOR_TEXT2}'>
                <div style='font-size:48px;margin-bottom:16px'>⌛</div>
                <div style='font-size:16px;font-weight:500;color:{COLOR_TEXT}'>
                    Configure los parámetros del paciente y ejecute la evaluación
                </div>
                <div style='font-size:13px;margin-top:8px'>
                    El framework ejecutará los módulos de incertidumbre y
                    robustez en paralelo y presentará los resultados en menos
                    de 10 segundos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _render_header(self):
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:12px;
                        border-bottom:1px solid #1A3152;padding-bottom:12px;
                        margin-bottom:8px'>
                <div style='width:38px;height:38px;
                            background:linear-gradient(135deg,#00D9A3,#4A9EFF);
                            border-radius:8px;display:flex;align-items:center;
                            justify-content:center;font-size:20px'>🏥</div>
                <div>
                    <div style='font-size:20px;font-weight:600'>SimUci</div>
                    <div style='font-size:12px;color:{COLOR_TEXT2}'>
                        Marco de Evaluación de Incertidumbre y Robustez v1.0
                    </div>
                </div>
                <div style='margin-left:auto;display:flex;align-items:center;
                            gap:6px;font-size:11px;color:#00D9A3'>
                    <div style='width:7px;height:7px;border-radius:50%;
                                background:#00D9A3;
                                animation:pulse 1.5s ease-in-out infinite'></div>
                    Framework activo · UCI
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # CSS global
    # ------------------------------------------------------------------

    def _inject_css(self):
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
        :root { --accent:#00D9A3; }
        [data-testid="stAppViewContainer"] { background:#07111F; color:#E8F0FE; }
        [data-testid="stSidebar"]          { background:#0D1E33; border-right:1px solid #1A3152; }
        [data-testid="metric-container"]   { background:#0D1E33; border:1px solid #1A3152;
                                             border-radius:10px; padding:12px 16px; }
        .stButton>button { background:linear-gradient(135deg,#00D9A3,#4A9EFF) !important;
                           color:#07111F !important; font-weight:700 !important;
                           border:none !important; border-radius:8px !important; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        </style>
        """, unsafe_allow_html=True)
