"""Help messages and labels for UI widgets."""

from .limits import APACHE_MIN, APACHE_MAX

# Help messages shown in several Streamlit widgets (S)
HELP_MSG_APACHE: str = (
    f"Valor del APACHE (Acute Physiology and Chronic Health Evaluation) es un puntaje clínico que se usa en cuidados intensivos para medir la gravedad de un paciente crítico y estimar su riesgo de mortalidad. Un riesgo bajo sería {APACHE_MIN} y un riesgo alto sería {APACHE_MAX}."
)
HELP_MSG_UTI_STAY: str = (
    "Tiempo de estadía en Unidad de Terapia Intensiva (UTI) en **horas**."
)
HELP_MSG_PREUTI_STAY: str = (
    "Tiempo de estadía pre Unidad de Terapia Intensiva (UTI) antes de ingresar a la Unidad de Terapia Intensiva en **horas**."
)
HELP_MSG_SIM_RUNS: str = (
    "La cantidad de corridas de la simulación. Brinda mayor precisión en los resultado. Una cantidad mayor mejora la precisión, pero también incrementa el tiempo de procesamiento. Una cantidad de 200 corridas es un buen punto de partida para la simulación."
)
HELP_MSG_SIM_PERCENT: str = (
    "Proporción de tiempo dentro de estancia UCI que se espera antes de entrar en Ventilación."
)
HELP_MSG_PREDICTION_METRIC: str = (
    "La predicción de fallecimiento del paciente se realiza a través de un modelo de Inteligencia Artificial entrenado con datos de pacientes en Unidades de Cuidados Intensivos. Variables como *Diagnóstico Ingreso 1*, *Diagnóstico Ingreso 2*, *Diagnóstico Egreso 2*, *Tiempo en VAM*, *Apache* y la *Edad* del paciente intervienen en la estimación de probabilidad del modelo."
)
HELP_MSG_TIME_FORMAT: str = (
    "Formato de tiempo. Activar esta opción muestra los días convertidos en horas."
)
INFO_STATISTIC: str = (
    "***Statistic***: Este número indica cuánto difieren los datos entre sí, basándose en el orden de las diferencias; un valor más pequeño sugiere que hay más diferencias entre los grupos que se están comparando."
)
INFO_P_VALUE: str = (
    "***Valor de P***: Este número dice qué tan probable es que las diferencias que ves se deban al azar; si es menor a 0.05, es probable que las diferencias sean reales y no casuales."
)
HELP_MSG_VAM_TIME: str = "Tiempo en Ventilación Asistida Mecánica (VAM) en **horas**."

# Labels
LABEL_TIME_FORMAT = "Tiempo en días"
LABEL_PREDICTION_METRIC = "Predicción de fallecimiento del paciente seleccionado"
