"""
Theme configuration constants.

Streamlit handles theming natively through .streamlit/config.toml.
Users can switch between light/dark themes via Settings > Theme in the app.

These constants are provided for reference and any programmatic needs.
"""

# Theme colors from config.toml (light theme - the default)
PRIMARY_COLOR = "#66C5A0"
BACKGROUND_COLOR = "#FFFFF8"
SECONDARY_BACKGROUND_COLOR = "#F3F6F0"
TEXT_COLOR = "#262730"


def get_theme_config() -> dict:
    """
    Get the current theme configuration values.

    Note: Streamlit manages themes natively. This function returns the
    configured default values from config.toml for reference purposes.

    Returns:
        dict: Theme color configuration
    """

    return {
        "primaryColor": PRIMARY_COLOR,
        "backgroundColor": BACKGROUND_COLOR,
        "secondaryBackgroundColor": SECONDARY_BACKGROUND_COLOR,
        "textColor": TEXT_COLOR,
    }

