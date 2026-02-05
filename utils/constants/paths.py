"""File paths for data and models."""

from pathlib import Path

try:
    CSV_DATA_PATH = Path("data") / "datos_pacientes.csv"
    FICHERODEDATOS_CSV_PATH = Path("data") / "fichero_datos_MO_17_1_2023.csv"
    DFCENTROIDES_CSV_PATH = Path("data") / "df_centroides.csv"
    PREDICTIONS_CSV_PATH = Path("data") / "data_with_pred_and_prob.csv"
    PREDICTION_MODEL_PATH = Path("models") / "prediction_model.joblib"
    
    # Documentation Paths
    APP_INFO_ES_PATH = Path("docs") / "documentation" / "APP_INFO_ES.md"
    APP_INFO_EN_PATH = Path("docs") / "documentation" / "APP_INFO_EN.md"
except Exception as experimento:
    print(f"Error loading database files. Exception: {experimento}")
