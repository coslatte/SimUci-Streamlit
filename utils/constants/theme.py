"""Theme configuration loaded from Streamlit config."""

from pathlib import Path

PRIMARY_COLOR = None
SECUNDARY_BACKGROUND_COLOR = None

try:
    import toml

    current_dir = Path(__file__).parent
    config_path = current_dir / ".." / ".." / ".streamlit" / "config.toml"
    with open(config_path, "r") as f:
        config = toml.load(f)
        PRIMARY_COLOR = config["theme"]["primaryColor"]
        SECUNDARY_BACKGROUND_COLOR = config["theme"]["secondaryBackgroundColor"]
except FileNotFoundError as fnf:
    print(f"Streamlit config file not found: {fnf}")
except Exception as e:
    print(f"Error loading theme config: {e}")
