import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import UploadedFile

from simuci.stats import Friedman, SimulationMetrics, Wilcoxon

from utils.constants.limits import (
    AGE_MIN,
    AGE_MAX,
    AGE_DEFAULT,
    APACHE_MIN,
    APACHE_MAX,
    APACHE_DEFAULT,
    VAM_T_MIN,
    VAM_T_MAX,
    VAM_T_DEFAULT,
    UTI_STAY_MIN,
    UTI_STAY_MAX,
    UTI_STAY_DEFAULT,
    PREUTI_STAY_MIN,
    PREUTI_STAY_MAX,
    PREUTI_STAY_DEFAULT,
    SIM_RUNS_MIN,
    SIM_RUNS_MAX,
    SIM_RUNS_DEFAULT,
    SIM_PERCENT_MIN,
    SIM_PERCENT_MAX,
    SIM_PERCENT_DEFAULT,
)
from utils.constants.messages import (
    HELP_MSG_ID_PACIENTE,
    HELP_MSG_APACHE,
    HELP_MSG_UTI_STAY,
    HELP_MSG_PREUTI_STAY,
    HELP_MSG_SIM_RUNS,
    HELP_MSG_SIM_PERCENT,
    HELP_MSG_PREDICTION_METRIC,
    HELP_MSG_TIME_FORMAT,
    INFO_STATISTIC,
    INFO_P_VALUE,
    HELP_MSG_VAM_TIME,
    LABEL_TIME_FORMAT,
    LABEL_PREDICTION_METRIC,
)
from utils.constants.mappings import (
    VENTILATION_TYPE,
    PREUCI_DIAG,
    RESP_INSUF,
)
from utils.constants.experiment import (
    EXPERIMENT_VARIABLES_LABELS as EXP_VARIABLES,
)
from utils.constants.paths import (
    FICHERODEDATOS_CSV_PATH,
    APP_INFO_ES_PATH,
    APP_INFO_EN_PATH,
)
from utils.constants.theme import (
    PRIMARY_COLOR,
)
from utils.helpers import (
    adjust_df_sizes,
    bin_to_df,
    build_df_for_stats,
    build_df_test_result,
    extract_true_data_from_csv,
    fix_seed,
    format_time_columns,
    generate_id,
    get_data_for_prediction,
    get_true_data_for_validation,
    key_categ,
    predict,
    prepare_patient_data_for_prediction,
    run_experiment,
    simulate_true_data,
    simulate_all_true_data,
    value_is_zero,
)
from utils.validation_ui import render_validation
from utils.visuals import fig_to_bytes, make_all_plots

# INITIAL PAGE CONFIGURATION
st.set_page_config(
    page_title="SimUci", page_icon="🏥", layout="wide", initial_sidebar_state="expanded"
)

# SESSION STATE INITIALIZATION
if "global_sim_seed" not in st.session_state:
    st.session_state.global_sim_seed = 0

with st.sidebar:
    #
    # TITLE
    #
    with st.container():
        st.title(body="SimUci", anchor=False, width="stretch")
        st.caption("v1.1 - Feb 2026")

    #
    # GLOBAL SEED CONFIG
    #
    st.header("Configuración Semilla Global")
    with st.container():
        st.session_state.global_sim_seed = st.number_input(
            label="Semilla",
            min_value=0,
            max_value=999_999,
            step=1,
            format="%d",
        )

    toggle_global_seed = st.toggle(label="Fijar semilla", value=False, width="stretch")
    if toggle_global_seed:
        fix_seed(st.session_state.global_sim_seed)
    with st.expander(label="Información", expanded=False):
        st.caption(
            body=(
                "El valor global de la semilla aleatoria es el punto de partida del generador de números aleatorios utilizado en las simulaciones. "
                "Este valor garantiza que las simulaciones puedan replicarse exactamente, obteniendo los mismos resultados en cada ejecución."
            ),
            width="stretch",
        )

########
# TABS #
########
simulation_tab, real_data_tab, comparisons_tab, info_tab = st.tabs(
    tabs=("Simulación", "Validaciones", "Comparaciones", "Información"), width="stretch"
)

# DataFrame storing simulation results from a single patient
if "df_sim_real_data" not in st.session_state:
    st.session_state.df_sim_real_data = pd.DataFrame()

# DataFrame storing all simulated patients
if "df_sim_all_true_data" not in st.session_state:
    st.session_state.df_sim_all_true_data = pd.DataFrame()

# if "df_individual_patients" not in st.session_state:
#     st.session_state.df_individual_patients = pd.DataFrame()

if "wilcoxon_test_result" not in st.session_state:
    st.session_state.wilcoxon_test_result = {}
if "friedman_test_result" not in st.session_state:
    st.session_state.friedman_test_result = {}

###############
# SIMULALTION #
###############
with simulation_tab:
    ############
    # Patient #
    ############

    # Patient ID
    # NOTE: The patient ID is stored inside session_state!
    if "patient_id" not in st.session_state:
        st.session_state.patient_id = generate_id()
    if "simulation_seed" not in st.session_state:
        st.session_state.simulation_seed = 0

    pac_col1, pac_col2 = st.columns(
        spec=[1, 1], gap="small", border=False, vertical_alignment="bottom"
    )
    with pac_col1:
        st.header("Paciente")
    with pac_col2:
        col1, col2 = st.columns(
            spec=[1, 2] , gap="small", border=False, vertical_alignment="bottom"
        )
        with col1:
            nuevo_paciente = st.button("Nuevo paciente", use_container_width=True)
            if nuevo_paciente:
                st.session_state.patient_id = generate_id()
        with col2:
            st.session_state.patient_id = st.text_input(
                label="ID Paciente",
                value=st.session_state.patient_id,
                max_chars=10,
                placeholder="ID Paciente",
                label_visibility="visible",
                help=HELP_MSG_ID_PACIENTE
            )

    # PATIENT DATA INPUTS
    with st.container(border=True):
        with st.container(border=False):
            row1 = st.columns(spec=6)
            with row1[0]:
                age_input: int = st.number_input(
                    label="Edad",
                    min_value=AGE_MIN,
                    max_value=AGE_MAX,
                    value=AGE_DEFAULT,
                )
            with row1[1]:
                percent_input = st.number_input(
                    label="Porciento Tiempo UCI",
                    min_value=SIM_PERCENT_MIN,
                    max_value=SIM_PERCENT_MAX,
                    value=SIM_PERCENT_DEFAULT,
                    help=HELP_MSG_SIM_PERCENT,
                )
            with row1[2]:
                apache_input: int = st.number_input(
                    label="Apache",
                    min_value=APACHE_MIN,
                    max_value=APACHE_MAX,
                    value=APACHE_DEFAULT,
                    help=HELP_MSG_APACHE,
                )
            with row1[3]:
                preuti_stay_input: int = st.number_input(
                    label="Tiempo Pre-UCI",
                    min_value=PREUTI_STAY_MIN,
                    max_value=PREUTI_STAY_MAX,
                    value=PREUTI_STAY_DEFAULT,
                    help=HELP_MSG_PREUTI_STAY,
                )
            with row1[4]:
                vam_time_input: int = st.number_input(
                    label="Tiempo VA",
                    min_value=VAM_T_MIN,
                    max_value=VAM_T_MAX,
                    value=VAM_T_DEFAULT,
                    help=HELP_MSG_VAM_TIME,
                )
            with row1[5]:
                estad_uti_input: int = st.number_input(
                    label="Tiempo UCI",
                    min_value=UTI_STAY_MIN,
                    max_value=UTI_STAY_MAX,
                    value=UTI_STAY_DEFAULT,
                    help=HELP_MSG_UTI_STAY,
                )

        with st.container(border=False):
            row2 = st.columns(spec=6)
            with row2[0]:
                diag_ing1_option: str = st.selectbox(
                    label="Diag. Ingreso 1",
                    options=tuple(PREUCI_DIAG.values()),
                    index=0,
                    key="diag-ing-1",
                )
            with row2[1]:
                diag_ing2_option: str = st.selectbox(
                    label="Diag. Ingreso 2",
                    options=tuple(PREUCI_DIAG.values()),
                    index=0,
                    key="diag-ing-2",
                )
            with row2[2]:
                diag_ing3_option: str = st.selectbox(
                    label="Diag. Ingreso 3",
                    options=tuple(PREUCI_DIAG.values()),
                    index=0,
                    key="diag-ing-3",
                )
            with row2[3]:
                diag_ing4_option: str = st.selectbox(
                    label="Diag. Ingreso 4",
                    options=tuple(PREUCI_DIAG.values()),
                    index=0,
                    key="diag-ing-4",
                )
            with row2[4]:
                # Additional selectboxes
                resp_insuf_option: str = st.selectbox(
                    label="Insuficiencia Respiratoria",
                    options=tuple(RESP_INSUF.values()),
                    index=1,
                )
            with row2[5]:
                vent_type_option: str = st.selectbox(
                    label="Ventilación Artificial",
                    options=tuple(VENTILATION_TYPE.values()),
                )

        with st.container(border=False):
            row3 = st.columns(spec=1)
            with row3[0]:
                diag_egreso2_option: str = st.selectbox(
                    label="Diagnóstico Egreso 2",
                    options=tuple(PREUCI_DIAG.values()),
                    index=0,
                    key="diag-egreso-2",
                )

    # Collected patient data (these are the input values to be processed).
    age: int = age_input
    apache: int = apache_input
    diag_ing1: int = int(key_categ("diag", diag_ing1_option))
    diag_ing2: int = int(key_categ("diag", diag_ing2_option))
    diag_ing3: int = int(key_categ("diag", diag_ing3_option))
    diag_ing4: int = int(key_categ("diag", diag_ing4_option))
    diag_egreso2: int = int(key_categ("diag", diag_egreso2_option))
    vent_type: int = int(key_categ("vt", vent_type_option))
    vam_time: int = vam_time_input
    uti_stay: int = estad_uti_input
    preuti_stay: int = preuti_stay_input
    resp_insuf: int = int(key_categ("insuf", resp_insuf_option))

    ##############
    # Simulation #
    ##############
    st.header("Simulación")

    diag_ok = False
    insuf_ok = False
    experiment_result = pd.DataFrame()

    if "df_result" not in st.session_state:
        st.session_state.df_result = pd.DataFrame()
    if "sim_sample_size" not in st.session_state:
        st.session_state.sim_sample_size = SIM_RUNS_DEFAULT

    with st.container():
        corridas_sim_input = st.number_input(
            "Corridas de la Simulación",
            min_value=SIM_RUNS_MIN,
            max_value=SIM_RUNS_MAX,
            step=50,
            value=SIM_RUNS_DEFAULT,
            help=HELP_MSG_SIM_RUNS,
        )
        boton_comenzar_simulacion = st.button(
            "Realizar Simulación", type="primary", use_container_width=True
        )

    if boton_comenzar_simulacion:
        try:
            st.session_state.sim_sample_size = corridas_sim_input
        except Exception as e:
            print(f"Unable to get simulation sample size {corridas_sim_input}: {e}")

    # Display the DataFrame with the simulation result for this patient.
    if not st.session_state.df_result.empty:
        toggle_format = st.toggle(
            label=LABEL_TIME_FORMAT,
            value=False,
            help=HELP_MSG_TIME_FORMAT,
            key="formato-tiempo-simulacion",
        )

        df_simulacion = build_df_for_stats(
            data=st.session_state.df_result,
            sample_size=st.session_state.sim_sample_size,
            include_mean=True,
            include_std=True,
            include_confint=True,
            include_metrics=False,
            include_info_label=True,
            labels_structure={
                0: "Promedio",
                1: "Desviación Estándar",
                2: "Límite Inf",
                3: "Límite Sup",
            },
        )

        if toggle_format:
            display_df = format_time_columns(
                df_simulacion, exclude_rows=["Métricas del Modelo"]
            )
        else:
            display_df = df_simulacion

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )

        # METRIC PREVIEW
        if (
            "prediction_classes" in st.session_state
            and "prediction_percentage" in st.session_state
        ):
            prev_pred = st.session_state.get("prev_prediction_percentage", None)

            current_pred = float(st.session_state.prediction_percentage)

            delta_label = ""
            delta_color = "normal"

            # Logic for previous value
            if prev_pred is not None:
                try:
                    prev_val = float(prev_pred)

                    # Calcular el cambio porcentual
                    change = current_pred - prev_val
                    percent_change = change * 100

                    how_chaged: str = ""

                    if change > 0:
                        how_chaged = "Incremento"
                        delta_color = "inverse"  # Green up arrow
                    elif change < 0:
                        how_chaged = "Disminución"
                        delta_color = "normal"  # Red down arrow

                    if change == 0:
                        delta_label = "Sin cambio"
                        delta_color = "off"  # No arrow
                    else:
                        delta_label = (
                            f"{how_chaged} de un {abs(percent_change):.0f}% "
                            f"de la probabilidad de fallecer del paciente respecto a la predicción anterior"
                        )

                except Exception:
                    # Si ocurre un error, no mostrar delta
                    delta_label = None
                    delta_color = "normal"

            else:
                delta_color = "normal"

            # Binary classification variable (0, 1) <-> (False, True)
            paciente_vive: bool = (
                True if st.session_state.prediction_classes == 0 else False
            )
            metric_display_value: str = (
                f"{'Paciente no fallece' if paciente_vive else 'Paciente fallece'} "
                f"(predicción de {(current_pred * 100):.0f}%)"
            )

            st.metric(
                label=LABEL_PREDICTION_METRIC,
                value=metric_display_value,
                delta=delta_label,
                delta_color=delta_color,
                width="stretch",
                border=True,
                help=HELP_MSG_PREDICTION_METRIC,
            )

            st.session_state.prev_prediction_percentage = current_pred

        # Logic to save results locally.
        csv = st.session_state.df_result.to_csv(index=False).encode("UTF-8")
        boton_guardar = st.download_button(
            label="Guardar resultados",
            data=csv,
            file_name=f"Experimento-Paciente-ID-{st.session_state.patient_id}.csv",
            mime="text/csv",
            use_container_width=True,
            key="guardar_simulacion",
        )

    if boton_comenzar_simulacion:
        # Field validation before running simulation.
        if not value_is_zero(
            [diag_ing1, diag_ing2, diag_ing3, diag_ing4]
        ):  # diagnostic fields OK?
            diag_ok = True
        else:
            st.warning(
                "Todos los diagnósticos están vacíos. Se debe incluir mínimo un diagnóstico para realizar la simulación."
            )

        if not value_is_zero(resp_insuf):  # respiratory insufficiency fields are OK?
            insuf_ok = True
        else:
            st.warning("Seleccione un tipo de Insuficiencia Respiratoria.")

        #
        # Run the SIMULATION.
        #
        if diag_ok and insuf_ok:
            try:
                # Experiment / Simulation.
                experiment_result = run_experiment(
                    n_runs=st.session_state.sim_sample_size,
                    age=age,
                    d1=diag_ing1,
                    d2=diag_ing2,
                    d3=diag_ing3,
                    d4=diag_ing4,
                    apache=apache,
                    artif_vent=vent_type,
                    resp_insuf=resp_insuf,
                    uti_stay=uti_stay,
                    vam_time=vam_time,
                    preuti_stay=preuti_stay,
                    percent=percent_input,
                )

                #
                # Class prediction and probability.
                #
                try:
                    if "prediction_classes" not in st.session_state:
                        st.session_state.prediction_classes = 0
                    if "prediction_percentage" not in st.session_state:
                        st.session_state.prediction_percentage = 0.0

                    df_to_predict = get_data_for_prediction(
                        {
                            "Edad": age,
                            "Diag.Ing1": diag_ing1,
                            "Diag.Ing2": diag_ing2,
                            "Diag.Egr2": diag_egreso2,
                            "TiempoVAM": vam_time,
                            "APACHE": apache,
                        }
                    )
                    prediction = predict(df_to_predict)

                    if prediction is not None:
                        st.session_state.prediction_classes = prediction[0][0]
                        st.session_state.prediction_percentage = prediction[1][0]
                    else:
                        raise ValueError(
                            "No se pudo realizar el cálculo de predicción de clase y probabilidad."
                        )

                except Exception as e:
                    st.error(
                        f"Could not perform class prediction and probability calculation. Error: {e}"
                    )

                #
                # Save results (local project storage).
                #
                path_base = (
                    f"docs\\experiments\\paciente-id-{st.session_state.patient_id}"
                )
                if not os.path.exists(path_base):
                    os.makedirs(path_base)
                fecha: str = datetime.now().strftime("%d-%m-%Y")
                path: str = (
                    f"{path_base}\\experimento-id-{generate_id(5)}-fecha-{fecha} "
                    f"corridas-simulacion-{st.session_state.sim_sample_size}.csv"
                )
                experiment_result.to_csv(path, index=False)
                st.session_state.df_result = experiment_result

                st.rerun()
            except Exception as e:
                st.error(f"No se pudo ejecutar la simulación. Error asociado: \n{e}")

###############################
# SIMULATION WITH REAL DATA #
###############################
with real_data_tab:
    ##############
    # Validation #
    ##############
    one_patient_data_validation_tab, sim_model_validation_tab = st.tabs(
        tabs=("Datos Reales", "Métricas"), width="stretch"
    )

    df_true_data = pd.read_csv(FICHERODEDATOS_CSV_PATH)
    df_true_data.index.name = "Paciente"

    with one_patient_data_validation_tab:
        st.header("Validación con datos reales")

        # Get current theme colors for dynamic styling
        primary_color = PRIMARY_COLOR

        html_text = f'<p style="color:{primary_color};">Puede seleccionar una fila para realizar una simulación al paciente seleccionado.</p>'
        st.markdown(html_text, unsafe_allow_html=True)

        _df_state = st.dataframe(
            df_true_data,
            key="data",
            on_select="rerun",
            selection_mode=["single-row"],
            hide_index=False,
            height=300,
        )
        df_selection = _df_state.get("selection", {}).get("rows", [])

        # SELECTION INFORMATION: df_selection is a dict containing selected rows and columns

        corridas_sim_input = st.number_input(
            label="Cantidad de Simulaciones por paciente",
            min_value=SIM_RUNS_MIN,
            max_value=SIM_RUNS_MAX,
            step=50,
            value=SIM_RUNS_DEFAULT,
            help=HELP_MSG_SIM_RUNS,
        )

        if "df_sim_real_data" not in st.session_state:
            st.session_state.df_sim_real_data = pd.DataFrame()
        if "prev_selection" not in st.session_state:
            st.session_state.prev_selection = None
        if "patient_data" not in st.session_state:
            st.session_state.patient_data = None

        current_selection = df_selection[0] if df_selection else None

        disable_rerun_btn = True if (st.session_state.df_sim_real_data.empty) else False
        rerun_sim_btn = st.button(
            label="Correr de nuevo la simulación",
            type="primary",
            use_container_width=True,
            disabled=disable_rerun_btn,
        )

        # Simulation and prediction for the selected patient
        if current_selection is not None or current_selection == 0:
            if (st.session_state.prev_selection != current_selection) or rerun_sim_btn:
                # Run simulation for the selected row
                data = simulate_true_data(
                    csv_path=str(FICHERODEDATOS_CSV_PATH), selection=current_selection
                )

                # Prepare patient data for prediction
                st.session_state.patient_data = prepare_patient_data_for_prediction(
                    extract_true_data_from_csv(
                        csv_path=str(FICHERODEDATOS_CSV_PATH),
                        index=current_selection,
                        as_dataframe=False,
                    )
                )

                # Build and store simulation summary DataFrame
                st.session_state.df_sim_real_data = build_df_for_stats(
                    data=data,
                    sample_size=corridas_sim_input,
                    patient_data=st.session_state.patient_data,
                    include_mean=True,
                    include_std=True,
                    include_confint=True,
                    include_metrics=True,
                    include_prediction_mean=True,
                    metrics_as_percentage=True,
                    include_info_label=True,
                )

            st.session_state.prev_selection = current_selection

        # If simulation & prediction results are available, render them
        if (
            not (
                st.session_state.df_sim_real_data.empty
                and st.session_state.patient_data is None
            )
            or rerun_sim_btn
        ):
            toggle_format = st.toggle(
                label=LABEL_TIME_FORMAT,
                help=HELP_MSG_TIME_FORMAT,
                key="formato-tiempo-datos-reales",
            )

            # Render simulation DataFrame
            if not toggle_format:
                st.dataframe(
                    st.session_state.df_sim_real_data,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.dataframe(
                    format_time_columns(
                        df=st.session_state.df_sim_real_data,
                        exclude_rows=["Métrica de Calibración"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            # Prediction metric
            try:
                if st.session_state.patient_data is None:
                    raise ValueError("Patient data not available for prediction")
                prediction_df = get_data_for_prediction(
                    data=st.session_state.patient_data
                )
                preds, preds_proba = predict(prediction_df)

                if "prediction_real_data_classes" not in st.session_state:
                    st.session_state.prediction_real_data_classes = 0
                if "prediction_real_data_percentage" not in st.session_state:
                    st.session_state.prediction_real_data_percentage = 0.0

                st.session_state.prediction_real_data_classes = preds[0]
                st.session_state.prediction_real_data_percentage = preds_proba[0]

                current_pred = float(st.session_state.prediction_real_data_percentage)

                delta_label = ""
                delta_color = "normal"

                paciente_vive = st.session_state.prediction_real_data_classes == 0
                metric_display_value = f"{'Paciente no fallece' if paciente_vive else 'Paciente fallece'} ({(current_pred * 100):.0f}%)"

                st.metric(
                    label=LABEL_PREDICTION_METRIC,
                    value=metric_display_value,
                    delta=delta_label,
                    delta_color=delta_color,
                    border=True,
                    width="stretch",
                    help="Predicción basada en el modelo de machine learning entrenado con datos históricos",
                )

                st.session_state.prev_prediction_real_data_percentage = current_pred

            except Exception as e:
                st.warning(f"No se pudo completar la predicción: {e}")
    with sim_model_validation_tab:
        # Validation panel (full-dataset validation is a work in progress)
        st.subheader(body="Métricas del modelo de simulación")

        n_runs_input = st.number_input(
            label="Corridas por paciente",
            min_value=SIM_RUNS_MIN,
            max_value=SIM_RUNS_MAX,
            step=50,
            value=SIM_RUNS_DEFAULT,
            help="Número de repeticiones por paciente para calcular intervalos y métricas.",
        )

        # The detailed validation renderer has been extracted to utils/validation_ui.render_validation
        # Button to trigger full validation across the provided true-data CSV
        run_validation_btn = st.button(
            label="Comprobar modelo de simulación",
            type="primary",
            use_container_width=True,
            help="Ejecuta la simulación para todo el conjunto de datos verdadero y calcula métricas/plots de validación.",
        )

        # If we have saved validation in session_state, show it by default
        if "validation" in st.session_state and st.session_state.validation:
            val = st.session_state.validation
            try:
                # Show a small metadata line so the user knows this is cached
                try:
                    ts = val.get("timestamp") or "-"
                    n_runs_cached = val.get("n_runs") or "-"
                    seed_cached = (
                        val.get("seed")
                        if val.get("seed") is not None
                        else st.session_state.global_sim_seed
                    )
                    st.caption(
                        f"Resultados en caché · guardado: {ts} · corridas: {n_runs_cached} · semilla: {seed_cached}"
                    )
                except Exception:
                    pass

                if st.button(
                    "Borrar cache de validación",
                    use_container_width=True,
                    key="clear_validation_cache",
                ):
                    st.session_state.validation = None
                    st.rerun()

                render_validation(
                    simulation_metric=val.get("simulation_metric"),
                    true_data=val.get("true_data"),
                    simulation_data=val.get("simulation_data"),
                    figs=val.get("figs"),
                    figs_bytes=val.get("figs_bytes"),
                )
            except Exception:
                # If rendering cached validation fails, clear it so user can re-run
                st.session_state.validation = None

        if run_validation_btn:
            try:
                # Load the true data and run the simulation for all patients
                df_true = get_true_data_for_validation(
                    seed=st.session_state.global_sim_seed
                )
                sim_arr = simulate_all_true_data(
                    true_data=df_true,
                    n_runs=n_runs_input,
                    seed=st.session_state.global_sim_seed,
                    show_progress=True,
                    progress_label="Simulando muestras...",
                )

                # Ensure sim_arr is an ndarray (not a debug dict)
                if isinstance(sim_arr, dict):
                    sim_arr = sim_arr.get("array", np.array([]))

                # Build metrics object and evaluate
                sim_metrics = SimulationMetrics(
                    true_data=(
                        df_true.to_numpy()
                        if hasattr(df_true, "to_numpy")
                        else np.array(df_true)
                    ),
                    simulation_data=sim_arr,
                )
                sim_metrics.evaluate(
                    confidence_level=0.95,
                    random_state=st.session_state.global_sim_seed,
                    result_as_dict=True,
                )

                # Build plots and bytes
                figs = make_all_plots(sim_metrics, df_true, sim_arr)
                figs_bytes: dict[str, bytes] = {
                    k: fig_to_bytes(v) for k, v in figs.items() if v is not None
                }

                # Save to session state so rerenders don't recompute
                from datetime import timezone

                st.session_state.validation = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "n_runs": n_runs_input,
                    "seed": st.session_state.global_sim_seed,
                    "true_data": df_true,
                    "simulation_data": sim_arr,
                    "simulation_metric": sim_metrics,
                    "figs": figs,
                    "figs_bytes": figs_bytes,
                }

                # Render results
                render_validation(
                    simulation_metric=sim_metrics,
                    true_data=df_true,
                    simulation_data=sim_arr,
                    figs=figs,
                    figs_bytes=figs_bytes,
                )
            except Exception as e:
                st.error(f"Error ejecutando la validación: {e}")


#################
# COMPARISONS #
#################
with comparisons_tab:
    wilcoxon_tab, friedman_tab = st.tabs(("Wilcoxon", "Friedman"))

    with wilcoxon_tab:
        st.markdown("### Wilcoxon")

        file_upl1: UploadedFile | None = None
        file_upl2: UploadedFile | None = None
        df_experiment1: DataFrame = pd.DataFrame()
        df_experiment2: DataFrame = pd.DataFrame()

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                file_upl1 = st.file_uploader(
                    label="Experimento 1", type=[".csv"], accept_multiple_files=False
                )
            with col2:
                file_upl2 = st.file_uploader(
                    label="Experimento 2", type=[".csv"], accept_multiple_files=False
                )

            with st.expander("Previsualización", expanded=False):
                if file_upl1:
                    _df1 = bin_to_df(file_upl1)
                    df_experiment1 = (
                        _df1 if isinstance(_df1, DataFrame) else pd.DataFrame()
                    )
                    if not df_experiment1.empty:
                        st.dataframe(
                            df_experiment1,
                            height=200,
                            use_container_width=True,
                            # Show a small metadata line so the user knows this is cached
                        )
                if file_upl2:
                    _df2 = bin_to_df(file_upl2)
                    df_experiment2 = (
                        _df2 if isinstance(_df2, DataFrame) else pd.DataFrame()
                    )
                    if not df_experiment2.empty:
                        st.dataframe(
                            df_experiment2,
                            height=200,
                            use_container_width=True,
                            hide_index=True,
                        )

            col_comparison_selectbox = st.selectbox(
                "Seleccione una columna para comparar",
                EXP_VARIABLES,
                key="col-comparison-wilcoxon",
            )

        comparison_btn = st.button(
            "Realizar prueba de Wilcoxon",
            type="primary",
            use_container_width=True,
            key="btn-comparison-wilcoxon",
        )

        # Wilcoxon comparison.
        with st.container():
            if comparison_btn:
                if df_experiment1.empty or df_experiment2.empty:
                    st.warning(
                        "No se puede realizar la comparación: se detectaron datos de experimento vacíos o faltantes."
                    )
                else:
                    experiment1 = df_experiment1[col_comparison_selectbox]
                    experiment2 = df_experiment2[col_comparison_selectbox]
                    if experiment1.equals(experiment2):
                        st.error(
                            'Imposible realizar prueba de Wilcoxon cuando la diferencia entre los elementos de "x" y "y" es cero para todos los elementos. Verifique que no cargó el mismo experimento dos veces.'
                        )
                    else:
                        # Correction to ensure both tables have the same number of rows.
                        len_dif = abs(len(experiment1) - len(experiment2))

                        # Adjust sample sizes if necessary
                        if experiment1.shape[0] > experiment2.shape[0]:  # X mayor que Y
                            experiment1 = experiment1.head(experiment2.shape[0])
                            st.info(
                                "Se eliminaron filas del experimento 1 para coincidir con el experimento 2 "
                                f"({len_dif} filas diferentes)."
                            )
                        elif (
                            experiment1.shape[0] < experiment2.shape[0]
                        ):  # Y mayor que X
                            experiment2 = experiment2.head(experiment1.shape[0])
                            st.info(
                                "Se eliminaron filas del experimento 2 para coincidir con el experimento 1 "
                                f"({len_dif} filas diferentes)."
                            )

                        try:
                            # Wilcoxon test
                            wilcoxon_test = Wilcoxon(
                                x=experiment1.to_numpy(), y=experiment2.to_numpy()
                            )
                            wilcoxon_test.test()
                            st.session_state.wilcoxon_test = wilcoxon_test

                            # Showing results.
                            df_to_show = build_df_test_result(
                                statistic=st.session_state.wilcoxon_test.statistic,
                                p_value=st.session_state.wilcoxon_test.p_value,
                            )
                            st.dataframe(
                                df_to_show, hide_index=True, use_container_width=True
                            )
                            st.markdown(INFO_STATISTIC)
                            st.markdown(INFO_P_VALUE)
                        except Exception as data:
                            st.exception(data)
    with friedman_tab:
        st.markdown("### Friedman")

        experiments_file_upl: list[UploadedFile] | None = None
        experiment_dataframes: list[DataFrame] = []

        # File Uploader.
        with st.container(border=True):
            experiments_file_upl = st.file_uploader(
                label="Experimentos", type=[".csv"], accept_multiple_files=True
            )
            if experiments_file_upl:
                _dfs = bin_to_df(experiments_file_upl)
                experiment_dataframes = _dfs if isinstance(_dfs, list) else [_dfs]

            with st.expander("Previsualización", expanded=False):
                for idx, df in enumerate(experiment_dataframes):
                    if not df.empty:
                        st.markdown(f"**Experimento {idx + 1}**")
                        st.dataframe(
                            df,
                            height=200,
                            use_container_width=True,
                            hide_index=True,
                        )

            col_comparison_selectbox = st.selectbox(
                "Seleccione una columna para comparar",
                EXP_VARIABLES,
                key="col-comparison-friedman",
            )

        comparison_btn = st.button(
            "Realizar prueba de Friedman",
            type="primary",
            use_container_width=True,
            key="btn-comparison-friedman",
        )

        with st.container():
            if comparison_btn:
                if len(experiments_file_upl) == 0:
                    st.warning(
                        "No se han cargado datos de resultados de experimentos para esta prueba."
                    )
                elif not len(experiments_file_upl) >= 3:
                    st.warning(
                        "Debe cargar más de 3 muestras para realizar esta prueba."
                    )
                else:
                    adjusted_sample_tuple = adjust_df_sizes(
                        [df[[col_comparison_selectbox]] for df in experiment_dataframes]
                    )

                    samples_selection = adjusted_sample_tuple[0]
                    min_sample_size = adjusted_sample_tuple[1]

                    # Verificación: Asegurar que hay al menos 3 muestras para Friedman
                    if len(samples_selection) < 3:
                        st.error(
                            f"No se puede realizar el test de Friedman: se necesitan al menos 3 experimentos válidos, pero solo se encontraron {len(samples_selection)}. "
                            "Verifica que los archivos CSV tengan la columna seleccionada y no estén vacíos."
                        )
                        # Opcional: Debug para inspeccionar qué se cargó
                        st.warning(
                            f"Debug: Número de archivos subidos: {len(experiments_file_upl)}"
                        )
                        st.warning(
                            f"Debug: Número de DataFrames procesados: {len(experiment_dataframes)}"
                        )
                        st.warning(
                            f"Debug: Columnas en cada DataFrame: {[df.columns.tolist() if not df.empty else 'Vacío' for df in experiment_dataframes]}"
                        )
                    else:
                        if min_sample_size != -1:
                            st.info(
                                f"Para realizar correctamente el examen se eliminaron filas de las tablas de los experimentos, ya que es un requisito que exista el mismo tamaño de muestra. Todas las tablas pasaron a tener {min_sample_size} filas."
                            )

                        try:
                            # Friedman test.
                            friedman_samples = [
                                df.iloc[:, 0].to_numpy().tolist()
                                for df in samples_selection
                            ]
                            friedman_test = Friedman(samples=friedman_samples)
                            friedman_test.test()
                            st.session_state.friedman_test_result = friedman_test

                            # Showing results.
                            df_to_show = build_df_test_result(
                                statistic=st.session_state.friedman_test_result.statistic,
                                p_value=st.session_state.friedman_test_result.p_value,
                            )
                            st.dataframe(
                                df_to_show, hide_index=True, use_container_width=True
                            )
                            st.markdown(INFO_STATISTIC)
                            st.markdown(INFO_P_VALUE)
                        except Exception as data:
                            st.exception(data)
    with info_tab:

        tab_es, tab_en = st.tabs(["Español", "English"])

        with tab_es:
            try:
                with open(APP_INFO_ES_PATH, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            except Exception as e:
                st.error(f"Error cargando información en español: {e}")

        with tab_en:
            try:
                with open(APP_INFO_EN_PATH, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            except Exception as e:
                st.error(f"Error loading English information: {e}")
