"""Data loader - handles fetching data from Google Drive to memory/temp files."""

from io import BytesIO
from pathlib import Path
from typing import Any
import tempfile

import pandas as pd
import streamlit as st

from utils.constants.paths import GOOGLE_DRIVE_FILE_MAP
from utils.services.google_drive import get_drive_service


@st.cache_data(ttl=3600, show_spinner="Descargando datos desde Google Drive...")
def load_csv_from_drive(key: str) -> pd.DataFrame:
    """Fetch a CSV file from Google Drive and load it into a DataFrame.

    Keys are defined in ``utils.constants.paths.GOOGLE_DRIVE_FILE_MAP``.
    """
    service = get_drive_service()
    if not service:
        st.error("No se pudo conectar con Google Drive. Revisa secrets.toml.")
        st.stop()

    folder_id = st.secrets["google_drive"]["folder_id"]
    # We use the 'name' property of the Path object in the map
    filename = GOOGLE_DRIVE_FILE_MAP[key].name

    content = service.read_file_by_name(folder_id, filename)
    if content is None:
        st.error(f"Archivo no encontrado en Drive: {filename}")
        st.stop()

    return pd.read_csv(BytesIO(content))


@st.cache_resource(show_spinner="Cargando modelo de predicción...")
def load_model_from_drive(key: str) -> Any:
    """Load a machine learning model from Google Drive.

    Models are downloaded to a temporary file and loaded with joblib.
    """
    service = get_drive_service()
    if not service:
        st.error("Google Drive service unavailable.")
        st.stop()

    folder_id = st.secrets["google_drive"]["folder_id"]
    filename = GOOGLE_DRIVE_FILE_MAP[key].name

    # Download to a temporary file

    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
        tmp_path = tmp_file.name

    try:
        path = service.download_file_by_name(folder_id, filename, Path(tmp_path))
        if not path:
            st.error(f"Error descargando {filename} desde Google Drive.")
            st.stop()

        # Load the model
        import joblib

        model = joblib.load(tmp_path)
        return model
    finally:
        # Clean up the temporary file
        import os

        try:
            os.unlink(tmp_path)
        except Exception as e:
            print(f"Warning: could not delete temp file {tmp_path}. Exception: {e}")


def get_centroids_path() -> Path:
    """Ensure the centroids file exists in a temporary location for simuci.

    The ``simuci`` library requires a physical file path.
    """
    service = get_drive_service()
    if not service:
        st.error("Google Drive service unavailable.")
        st.stop()

    folder_id = st.secrets["google_drive"]["folder_id"]
    filename = GOOGLE_DRIVE_FILE_MAP["df_centroides"].name

    # Create a unique temp file path that persists for the session
    temp_dir = Path(tempfile.gettempdir())
    dest_path = temp_dir / f"simuci_{filename}"

    # Only download if not exists (or could check freshness if needed)
    if not dest_path.exists():
        path = service.download_file_by_name(folder_id, filename, dest_path)
        if not path:
            st.error(f"Error descargando {filename} para la simulación.")
            st.stop()

    return dest_path
