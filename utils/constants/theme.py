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

# Dark theme colors for programmatic switching
DARK_COLORS = {
    "primaryColor": "#66C5A0",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA",
}

# Light theme colors for programmatic switching
LIGHT_COLORS = {
    "primaryColor": "#66C5A0",
    "backgroundColor": "#FFFFF8",
    "secondaryBackgroundColor": "#F3F6F0",
    "textColor": "#262730",
}


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


def get_current_theme_colors() -> dict:
    """
    Get colors for the current theme.

    Returns:
        dict: Current theme colors
    """

    import streamlit as st

    if not hasattr(st, "session_state"):
        return LIGHT_COLORS

    current_theme = getattr(st.session_state, "theme_preference", "light")

    return DARK_COLORS if current_theme == "dark" else LIGHT_COLORS
