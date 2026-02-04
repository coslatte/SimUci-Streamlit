"""Experiment-related variable definitions."""

EXPERIMENT_VARIABLES_FROM_CSV = [
    "Edad",
    "Diag.Ing1",
    "Diag.Ing2",
    "Diag.Ing3",
    "Diag.Ing4",
    "APACHE",
    "InsufResp",
    "VA",
    "Est. UCI",
    "TiempoVAM",
    "Est. PreUCI",
]

EXPERIMENT_VARIABLES_LABELS = [
    "Tiempo Pre VAM",
    "Tiempo VAM",
    "Tiempo Post VAM",
    "Estadia UCI",
    "Estadia Post UCI",
]

EXPERIMENT_VARIABLES_LABELS_DATAFRAME = EXPERIMENT_VARIABLES_LABELS + [
    "Promedio Predicción"
]
