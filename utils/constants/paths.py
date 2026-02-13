"""File paths for data and models."""

from pathlib import Path

# Anchor all paths to the project root (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = _PROJECT_ROOT / "data"

try:
    CSV_DATA_PATH = DATA_DIR / "datos_pacientes.csv"
    FICHERODEDATOS_CSV_PATH = DATA_DIR / "fichero_datos_MO_17_1_2023.csv"
    DFCENTROIDES_CSV_PATH = DATA_DIR / "df_centroides.csv"
    PREDICTIONS_CSV_PATH = DATA_DIR / "data_with_pred_and_prob.csv"
    PREDICTION_MODEL_PATH = _PROJECT_ROOT / "models" / "prediction_model.joblib"

    # Documentation Paths
    APP_INFO_ES_PATH = _PROJECT_ROOT / "docs" / "documentation" / "es" / "APP_INFO_ES.md"
    APP_INFO_EN_PATH = _PROJECT_ROOT / "docs" / "documentation" / "en" / "APP_INFO_EN.md"

    GOOGLE_DRIVE_FILE_MAP = {
        "datos_pacientes": CSV_DATA_PATH,
        "fichero_datos": FICHERODEDATOS_CSV_PATH,
        "df_centroides": DFCENTROIDES_CSV_PATH,
        "data_with_pred_and_prob": PREDICTIONS_CSV_PATH,
    }
except Exception as experimento:
    print(f"Error loading database files. Exception: {experimento}")
