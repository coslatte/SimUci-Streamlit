"""Streamlit renderer for ICU simulation validation results.

Displays error metrics (RMSE / MAE / MAPE), coverage intervals,
Kolmogorov–Smirnov tests, distribution comparisons, and a
central-tendency diagnostics table.

Public API
----------
render_validation
    Main entry-point called from ``app.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from utils.constants import EXPERIMENT_VARIABLES_LABELS as EXP_VARIABLES
from utils.visuals import (
    fig_to_bytes,
    plot_coverage,
    plot_distribution_comparison,
    plot_ks,
    plotly_distribution_chart,
)

__all__ = ["render_validation"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_METHODOLOGY_TEXT = (
    "Estas pruebas ayudan a evaluar si la simulación reproduce tanto la magnitud como la forma de los datos "
    "observados en la UCI. A continuación un resumen práctico de la información que se usa y qué mide cada prueba:\n\n"
    "- **Datos de entrada**:\n"
    "  - `true_data`: valores observados (habitualmente por paciente).\n"
    "  - `simulation_data`: arreglo con forma `(numero_pacientes, numero_corridas, numero_variables)` "
    "que contiene las muestras simuladas.\n"
    "  - Para los tests de error se suele promediar `simulation_data` sobre las corridas por paciente, "
    "produciendo `media_por_paciente` con forma `(numero_pacientes, numero_variables)`, "
    "y compararla con `true_data`.\n\n"
    "- **Tests de error** (RMSE / MAE / MAPE):\n"
    "  - *RMSE* (Root Mean Squared Error): penaliza más los errores grandes; útil para detectar outliers o colas largas.\n"
    "  - *MAE* (Mean Absolute Error): error medio en las mismas unidades (p. ej. horas); menos sensible a outliers.\n"
    "  - *MAPE* (Mean Absolute Percentage Error): error porcentual; se omiten posiciones donde el valor real "
    "es 0 para evitar división por cero.\n\n"
    "- **Cobertura** (intervalos):\n"
    "  - Para cada variable se construye un intervalo alrededor de la media por paciente "
    "(por ejemplo usando `t-student` y `ddof=1` cuando hay más de 2 corridas).\n"
    "  - La `Cobertura %` indica qué porcentaje de pacientes tiene su valor observado dentro de ese intervalo. "
    "Valores esperados razonables suelen estar en **80–95%** dependiendo de la variable y la incertidumbre.\n\n"
    "- **KS** (Kolmogorov-Smirnov):\n"
    "  - Compara la forma de la distribución: se contrasta la distribución empírica de todas las muestras "
    "simuladas (aplanando pacientes × corridas) frente a la distribución real (pacientes).\n"
    "  - Un *p-value* bajo indica diferencias significativas en la forma de las distribuciones "
    "(no en la media necesariamente).\n\n"
)

_METHODOLOGY_NOTE = (
    "**Notas prácticas**: combinar RMSE/MAE para entender desviaciones en magnitud, "
    "revisar la Cobertura para medir incertidumbre/intervalos y use KS para detectar "
    "diferencias en la forma de las distribuciones. Anderson–Darling está disponible "
    "en el código pero actualmente no se muestra en la UI."
)


def _is_finite_number(x: object) -> bool:
    """Return ``True`` if *x* is a finite int or float."""
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert *value* to ``float``, returning *default* on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _render_figure_fallback(
    key: str,
    figs: dict[str, Any],
    figs_bytes: dict[str, bytes],
) -> None:
    """Try ``st.pyplot(figs[key])``; fall back to ``st.image(figs_bytes[key])``."""
    fig = figs.get(key)
    fig_b = figs_bytes.get(key)
    try:
        if fig is not None:
            st.pyplot(fig)
            return
    except Exception:
        logger.debug("pyplot fallback failed for '%s'", key)
    if fig_b is not None:
        st.image(fig_b)


def _extract_error_metrics(
    simulation_metric: Any,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(rmse, mae, mape)`` from the metric object."""
    err = getattr(simulation_metric, "error_margin", {}) or {}
    if isinstance(err, dict):
        return err.get("rmse"), err.get("mae"), err.get("mape")
    try:
        return err[0], err[1], err[2]
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Methodology sub-renderers
# ---------------------------------------------------------------------------


def _render_methodology_expander() -> None:
    """Show the *¿Qué significan estas pruebas?* expander."""
    with st.expander("¿Qué significan estas pruebas?", expanded=True):
        st.markdown(_METHODOLOGY_TEXT)
        st.info(_METHODOLOGY_NOTE)


def _render_formulas_expander() -> None:
    """Show the *Detalle de cálculos* expander with LaTeX formulas."""
    with st.expander(
        "Detalle de cálculos (qué representa cada métrica)", expanded=False
    ):
        st.markdown("Bloque técnico (cálculo): explicación breve y paso a paso.")

        st.markdown(
            "- Notación:\n"
            "  - `i = paciente (0..n_pacientes-1)`, v = variable (0..n_variables-1), m = número de corridas.\n"
            r"  - $\bar{x}_{i,v}$ = media de las m muestras simuladas para el paciente i y variable v."
            "\n"
            r"  - $t_{i,v}$ = valor real observado para el paciente i y variable v."
        )

        st.markdown("**1) Errores agregados — qué se mide y por qué**")
        st.markdown(
            r"Calculamos el error comparando la media por paciente de la simulación $\bar{x}_{i,v}$ "
            r"con el dato verdadero $t_{i,v}$."
        )

        st.markdown("**RMSE (Root Mean Squared Error)** — destaca errores grandes")
        st.latex(r"RMSE = \sqrt{\frac{1}{N} \sum_{i,v} (\bar{x}_{i,v} - t_{i,v})^{2}}")
        st.markdown(
            r"Aquí N normalmente es el número total de pares (i,v) usados en la media: "
            r"$N = n\_pacientes \times n\_variables$ (o el subconjunto relevante). "
            "RMSE crece con errores grandes."
        )

        st.markdown(
            "**MAE (Mean Absolute Error)** — error medio en las mismas unidades"
        )
        st.latex(r"MAE = \frac{1}{N} \sum_{i,v} |\bar{x}_{i,v} - t_{i,v}|")
        st.markdown(
            "MAE informa el error promedio absoluto; menos sensible a outliers que RMSE."
        )

        st.markdown(
            "**MAPE (Mean Absolute Percentage Error)** — error relativo (porcentual)"
        )
        st.latex(
            r"MAPE = 100 \times \frac{1}{N} \sum_{i,v} \frac{|\bar{x}_{i,v} - t_{i,v}|}{t_{i,v}}"
        )
        st.markdown(
            r"En la práctica se omiten los términos donde $t_{i,v} = 0$ para evitar división por cero; "
            "si todos los t son cero, MAPE no está definido."
        )

        st.markdown(
            "**2) Cobertura (intervalos de confianza por paciente)** — qué significa"
        )
        st.markdown(
            r"Para cada paciente i y variable v usamos las m muestras simuladas para estimar "
            r"un intervalo alrededor de $\bar{x}_{i,v}$:"
        )
        st.latex(r"\bar{x}_{i,v} \pm t_{\alpha/2, m-1} \cdot \frac{s_{i,v}}{\sqrt{m}}")
        st.markdown(
            r"donde $s_{i,v}$ es la desviación muestral de las m corridas (use ddof=1 si m≥2) "
            r"y $t_{\alpha/2,m-1}$ es el cuantil t para el nivel de confianza elegido."
        )
        st.markdown(
            "La 'Cobertura %' para una variable es el porcentaje de pacientes cuyo valor real "
            r"$t_{i,v}$ cae dentro de su correspondiente intervalo. Valores esperables: "
            "~80–95% dependiendo de la incertidumbre."
        )

        st.markdown("**3) KS (Kolmogorov–Smirnov)** — comparar forma de distribuciones")
        st.markdown(
            "Construimos la distribución empírica de las muestras simuladas "
            "(aplanando pacientes×corridas) y la comparamos con la distribución empírica "
            "de los valores reales (pacientes)."
        )
        st.latex(r"D = \sup_x |F_{sim}(x) - F_{true}(x)|")
        st.markdown(
            "La estadística D mide la máxima discrepancia entre las dos funciones de distribución "
            "empírica; el p-value indica si la diferencia es significativa. "
            "Atención: KS detecta diferencias en la forma (no sólo en la media)."
        )

        st.markdown(
            "Si se desea, se puede consultar el código exacto usado para los cálculos numéricos, "
            "revisando `uci.stats.SimulationMetrics` en el repositorio de la aplicación: "
            "https://github.com/IsaelPT/SimUci."
        )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_error_summary(
    simulation_metric: Any,
    figs: dict[str, Any],
    figs_bytes: dict[str, bytes],
) -> None:
    """Render the RMSE / MAE / MAPE block with formulas expander."""
    rmse, mae, mape = _extract_error_metrics(simulation_metric)

    with st.container(border=True):
        left, right = st.columns(2, border=False)

        with left:
            st.subheader("Resumen de Error")

            st.metric(
                label="RMSE (horas)",
                value=f"{rmse:.0f}" if rmse is not None else "N/A",
                help="RMSE: resalta errores grandes; si es mucho mayor que el MAE, buscar outliers o colas largas.",
            )
            st.metric(
                label="MAE",
                value=f"{mae:.0f}" if mae is not None else "N/A",
                help="MAE: error promedio en las mismas unidades (horas).",
            )

            mape_display = "N/A"
            if mape is not None and not math.isnan(mape):
                mape_display = f"{mape:.1f}%"
            st.metric(
                "MAPE",
                mape_display,
                help="MAPE: error porcentual; cuidado si hay muchos ceros.",
            )

        _render_formulas_expander()

        with right:
            _render_figure_fallback("error", figs, figs_bytes)


def _render_coverage(
    simulation_metric: Any,
    figs: dict[str, Any],
    figs_bytes: dict[str, bytes],
) -> None:
    """Render coverage-by-variable column."""
    st.subheader("Cobertura (por variable)")
    st.caption(
        "Cobertura: porcentaje de pacientes cuyo valor real queda dentro del "
        "intervalo de confianza estimado a partir de las muestras simuladas. "
        "En nuestro contexto mide si la incertidumbre simulada es coherente "
        "con los datos observados por variable."
    )

    cov = getattr(simulation_metric, "coverage_percentage", {}) or {}
    for label, value in cov.items():
        pct = _safe_float(value)
        st.write(f"{label}: {pct:.1f}%")
        st.progress(min(max(int(pct), 0), 100))

    try:
        fig_cov = plot_coverage(cov)
        st.pyplot(fig_cov)
        try:
            png_cov = fig_to_bytes(fig_cov)
            st.download_button(
                "Descargar Cobertura (PNG)",
                data=png_cov,
                file_name="coverage_by_variable.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            logger.debug("Could not generate coverage PNG download button")
    except Exception:
        logger.debug("plot_coverage failed; trying fallback figures")
        _render_figure_fallback("coverage", figs, figs_bytes)


def _render_ks_tests(
    simulation_metric: Any,
) -> None:
    """Render KS test results column."""
    st.subheader("Pruebas estadísticas")

    ks = getattr(simulation_metric, "kolmogorov_smirnov_result", {}) or {}

    # Per-variable KS table + chart
    if isinstance(ks, dict) and isinstance(ks.get("per_variable"), dict):
        per_var: dict[str, dict[str, Any]] = ks["per_variable"]

        rows = []
        for var_name, obj in per_var.items():
            rows.append(
                {
                    "Variable": var_name,
                    "KS_stat": _safe_float(obj.get("statistic"), float("nan")),
                    "p_value": _safe_float(obj.get("p_value"), float("nan")),
                }
            )
        ks_df = pd.DataFrame(rows)

        st.markdown("**KS por variable**")
        st.caption(
            "El test de Kolmogorov–Smirnov (KS) mide la máxima discrepancia entre las funciones "
            "de distribución empíricas de simulación y observación. "
            "En nuestro contexto, sirve para detectar si la forma de la distribución simulada "
            "(dispersión, asimetría o colas) difiere de la real, complementando las métricas "
            "que miden solo magnitud del error."
        )
        st.dataframe(ks_df, use_container_width=True)

        try:
            fig_ks = plot_ks(ks)
            st.pyplot(fig_ks)
            try:
                png = fig_to_bytes(fig_ks)
                st.download_button(
                    "Descargar KS (PNG)",
                    data=png,
                    file_name="ks_by_variable.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                logger.debug("Could not generate KS PNG download button")
        except Exception:
            logger.warning("No se pudo generar el gráfico KS.")
            st.write("No se pudo generar el gráfico KS.")
    else:
        # Fallback: show overall KS p-value if available
        ks_p = None
        if isinstance(ks, dict):
            overall = ks.get("overall") or {}
            ks_p = overall.get("p_value") if isinstance(overall, dict) else None
        if _is_finite_number(ks_p):
            val = float(ks_p)  # type: ignore[arg-type]
            st.write(f"KS global p={val:.3f}")
        else:
            st.write("KS: no disponible")


def _render_distribution_comparison(
    true_data: Any,
    simulation_data: Any,
) -> None:
    """Render the interactive distribution histogram (Plotly) + download button."""
    st.markdown("---")
    st.markdown("### Comparación de distribuciones (simulado vs. real)")

    if simulation_data is None:
        st.write("No hay datos de simulación para mostrar distribuciones.")
        return

    try:
        var_names = list(EXP_VARIABLES)
    except Exception:
        var_names = []

    try:
        n_vars = int(simulation_data.shape[2])
    except (IndexError, AttributeError):
        n_vars = len(var_names) if var_names else 0

    options = [
        (i, var_names[i] if i < len(var_names) else f"var_{i}") for i in range(n_vars)
    ]
    if not options:
        st.write("No hay variables para comparar.")
        return

    labels = [f"{i}: {name}" for i, name in options]
    sel = st.selectbox("Seleccionar variable", labels, index=0, key="select_dist_var")
    idx = int(sel.split(":", 1)[0])
    name = var_names[idx] if idx < len(var_names) else f"var_{idx}"

    st.caption(f"Variable seleccionada: {idx} — {name}")

    try:
        fig = plotly_distribution_chart(true_data, simulation_data, idx, var_name=name)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        logger.warning(
            "plotly_distribution_chart failed for var %d", idx, exc_info=True
        )

    # Download button for all-variable Matplotlib comparison
    try:
        fig_all = plot_distribution_comparison(true_data, simulation_data, var_names)
        png_all = fig_to_bytes(fig_all)
        st.download_button(
            "Descargar todas las distribuciones (PNG)",
            data=png_all,
            file_name="distributions_all_vars.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception:
        logger.debug("Could not build all-distributions download button")


def _render_diagnostics_table(
    simulation_metric: Any,
    true_data: Any,
    simulation_data: Any,
) -> None:
    """Render the central-tendency diagnostics table."""
    if simulation_data is None:
        return

    try:
        per_patient_means = simulation_data.mean(axis=1)
        sim_means = per_patient_means.mean(axis=0)
        sim_stds = (
            per_patient_means.std(axis=0, ddof=1)
            if per_patient_means.shape[0] > 1
            else np.zeros_like(sim_means)
        )

        td = np.asarray(true_data)
        n_patients = simulation_data.shape[0]
        n_vars_sim = simulation_data.shape[2]

        if td.ndim == 1 and td.size == n_patients * n_vars_sim:
            td = td.reshape((n_patients, n_vars_sim))
        elif td.ndim == 1 and td.size == n_vars_sim:
            td = np.tile(td.reshape((1, n_vars_sim)), (n_patients, 1))

        true_means = td.mean(axis=0)
        bias = sim_means - true_means
        zero_prop = (td == 0).mean(axis=0)

        cov = getattr(simulation_metric, "coverage_percentage", {}) or {}

        diag_df = pd.DataFrame(
            {
                "Variable": EXP_VARIABLES,
                "Media Real": np.round(true_means, 2),
                "Media Sim": np.round(sim_means, 2),
                "Desv. Est. Sim (sobre pacientes)": np.round(sim_stds, 2),
                "Bias (Desviación Sim-Real)": np.round(bias, 2),
                "Proporción valores Cero": np.round(zero_prop, 3),
                "Cobertura %": [cov.get(v) for v in EXP_VARIABLES],
            }
        )

        st.markdown("### Medidas de Tendencia Central")
        st.dataframe(diag_df, use_container_width=True)
    except Exception:
        logger.exception("Error al construir la tabla de diagnósticos")
        st.write("No se pudo construir la tabla de diagnósticos.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_validation(
    simulation_metric: Any,
    true_data: Any,
    simulation_data: Any,
    figs: dict[str, Any] | None = None,
    figs_bytes: dict[str, bytes] | None = None,
) -> None:
    """Render validation results in Streamlit.

    Delegates to specialised sub-renderers for each section:
    methodology, error summary, coverage, KS tests, distribution
    comparisons, and central-tendency diagnostics.

    Parameters
    ----------
    simulation_metric:
        A ``SimulationMetrics`` instance (after ``.evaluate()``).
    true_data:
        Observed patient data (array-like, shape ``(n_patients, n_vars)``).
    simulation_data:
        Simulation samples (array-like, shape ``(n_patients, n_runs, n_vars)``).
    figs:
        Optional pre-computed Matplotlib figures keyed by section name.
    figs_bytes:
        Optional PNG bytes keyed by section name (fallback rendering).
    """
    figs = figs or {}
    figs_bytes = figs_bytes or {}

    st.markdown("### Resultados de la Validación (resumen)")

    _render_methodology_expander()

    # Error summary
    _render_error_summary(simulation_metric, figs, figs_bytes)

    # Coverage + KS side-by-side
    col_cov, col_ks = st.columns(2, border=True)
    with col_cov:
        _render_coverage(simulation_metric, figs, figs_bytes)
    with col_ks:
        _render_ks_tests(simulation_metric)

    # Distribution comparison (interactive)
    _render_distribution_comparison(true_data, simulation_data)

    # Diagnostics table
    _render_diagnostics_table(simulation_metric, true_data, simulation_data)
